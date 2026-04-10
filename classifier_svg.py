"""Toy experiment: train a tiny MLP to classify whether a 2D point is inside
any shape of an SVG.

The ground truth oracle is a high-resolution rasterization of the SVG: a
point is "inside" iff its nearest pixel is non-background. This is the same
even-odd fill rule ray casting would give for polygons, but also handles
curved paths that exist in the dataset.

We work in the normalized canvas space [0, 1]^2 (with x to the right and y
downward, matching screen / raster coordinates).
"""

import io
import math
import os

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from PIL import Image

PLOTS_DIR = "classifier_plots"
PARAMS_DIR = "classifier_parameters"


def load_one_svg(skip=0):
    """Fetch one raw SVG XML + its description from the dataset."""
    ds = load_dataset(
        "xingxm/SVGX-SFT-1M",
        split="train",
        data_files="SVGX_SFT_GEN_51k.json",
        streaming=True,
    )
    for i, item in enumerate(ds):
        svg = item.get("output", "")
        if not svg.strip().startswith("<svg"):
            continue
        if i < skip:
            continue
        return svg, item.get("input", "")
    raise RuntimeError("No SVG found in dataset")


def load_svgs(n, skip=0):
    """Fetch raw (svg_xml, description) pairs from the dataset.

    If n is None, returns every distinct SVG in the dataset. Otherwise stops
    after n and raises if fewer are available.

    The dataset is SFT-style: the same SVG often appears in several consecutive
    entries paired with different input prompts. We dedupe on the SVG body so
    each returned sample is a distinct image.
    """
    ds = load_dataset(
        "xingxm/SVGX-SFT-1M",
        split="train",
        data_files="SVGX_SFT_GEN_51k.json",
        streaming=True,
    )
    out = []
    seen = set()
    for i, item in enumerate(ds):
        svg = item.get("output", "")
        if not svg.strip().startswith("<svg"):
            continue
        if i < skip:
            continue
        key = svg.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((svg, item.get("input", "")))
        if n is not None and len(out) >= n:
            return out
    if n is not None and len(out) < n:
        raise RuntimeError(f"Only found {len(out)} SVGs, wanted {n}")
    return out


def rasterize_svg_mask(svg_xml, size):
    """Rasterize the SVG and return a boolean (H, W) mask of 'inside' pixels.

    A pixel is considered inside if it is sufficiently opaque; if the SVG has
    no transparency we fall back to non-white detection.
    """
    png = cairosvg.svg2png(
        bytestring=svg_xml.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    arr = np.array(img)
    alpha = arr[..., 3]
    if alpha.max() == 0:
        # Fully transparent render -> fall back to non-white on RGB
        rgb = arr[..., :3]
        mask = rgb.min(axis=-1) < 250
    elif alpha.min() == 255:
        # No alpha information -> non-white
        rgb = arr[..., :3]
        mask = rgb.min(axis=-1) < 250
    else:
        mask = alpha > 128
    return mask


def render_svg_to_image(svg_xml, size):
    """Rasterize an SVG to a PIL RGB image composited on a white background."""
    png = cairosvg.svg2png(
        bytestring=svg_xml.encode("utf-8"),
        output_width=size,
        output_height=size,
    )
    svg_img = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", svg_img.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, svg_img).convert("RGB")


def mask_lookup(mask_np, xy):
    """Look up ground-truth labels for points in [0, 1]^2.

    xy: (N, 2) tensor on any device with columns (x, y).
    Returns: (N,) float32 tensor of 0/1 labels on xy's device.
    """
    H, W = mask_np.shape
    device = xy.device
    u = (xy[:, 0].clamp(0.0, 1.0) * (W - 1)).long().cpu().numpy()
    v = (xy[:, 1].clamp(0.0, 1.0) * (H - 1)).long().cpu().numpy()
    labels = mask_np[v, u].astype(np.float32)
    return torch.from_numpy(labels).to(device)


class PointClassifier(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, xy):
        # Input is expected in [-1, 1] to keep activations well-scaled.
        return self.net(xy).squeeze(-1)


def train(model, mask, steps, batch_size, lr, device, wandb_prefix=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    losses = []
    for step in range(steps):
        xy = torch.rand(batch_size, 2, device=device)
        labels = mask_lookup(mask, xy)
        logits = model(xy * 2.0 - 1.0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean().item()
            print(f"step {step:5d}  loss {loss.item():.4f}  acc {acc:.3f}")
            if wandb_prefix is not None:
                wandb.log({
                    f"{wandb_prefix}/loss": loss.item(),
                    f"{wandb_prefix}/acc": acc,
                    f"{wandb_prefix}/step": step,
                })
    return model, losses


def evaluate_grid(model, size, device="cpu"):
    ys = torch.linspace(0.0, 1.0, size, device=device)
    xs = torch.linspace(0.0, 1.0, size, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xy * 2.0 - 1.0))
    return probs.view(size, size).cpu().numpy()


class BatchedPointClassifier(nn.Module):
    """B independent point classifiers trained in parallel via bmm.

    Parameters have a leading "model" dimension. Forward expects xy of shape
    (B, N, 2) and returns logits of shape (B, N), where each model i sees its
    own slice of points and computes its own output independently.
    """

    def __init__(self, n_models, hidden):
        super().__init__()
        self.n_models = n_models
        self.hidden = hidden
        self.W1 = nn.Parameter(torch.empty(n_models, 2, hidden))
        self.b1 = nn.Parameter(torch.empty(n_models, hidden))
        self.W2 = nn.Parameter(torch.empty(n_models, hidden, hidden))
        self.b2 = nn.Parameter(torch.empty(n_models, hidden))
        self.W3 = nn.Parameter(torch.empty(n_models, hidden, 1))
        self.b3 = nn.Parameter(torch.empty(n_models, 1))
        # Match nn.Linear's default init: kaiming_uniform on weights, uniform
        # bias with bound = 1/sqrt(fan_in).
        for W, b in [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3)]:
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            fan_in = W.shape[1]
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    def forward(self, xy):
        # xy: (B, N, 2) in [-1, 1]
        h = torch.bmm(xy, self.W1) + self.b1.unsqueeze(1)
        h = F.relu(h)
        h = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)
        h = F.relu(h)
        h = torch.bmm(h, self.W3) + self.b3.unsqueeze(1)
        return h.squeeze(-1)  # (B, N)


def batched_mask_lookup(masks_tensor, xy):
    """Look up labels in B masks at points in [0, 1]^2.

    masks_tensor: (B, H, W) bool/uint8 tensor on device.
    xy: (B, N, 2) float tensor on the same device.
    Returns: (B, N) float32 tensor of 0/1 labels.
    """
    B, H, W = masks_tensor.shape
    device = masks_tensor.device
    u = (xy[:, :, 0].clamp(0.0, 1.0) * (W - 1)).long()
    v = (xy[:, :, 1].clamp(0.0, 1.0) * (H - 1)).long()
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(u)
    return masks_tensor[batch_idx, v, u].float()


def batched_train(model, masks_tensor, steps, batch_size, lr, wandb_prefixes=None):
    """Train a BatchedPointClassifier on B masks in parallel.

    masks_tensor: (B, H, W) bool tensor on the same device as the model.
    batch_size is per-model (each model sees this many random points per step).
    Returns the model and a list of B per-model loss histories.
    """
    device = masks_tensor.device
    B = masks_tensor.shape[0]
    assert model.n_models == B, f"model has {model.n_models} heads but got {B} masks"

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    losses_per_model = [[] for _ in range(B)]

    for step in range(steps):
        xy = torch.rand(B, batch_size, 2, device=device)
        labels = batched_mask_lookup(masks_tensor, xy)
        logits = model(xy * 2.0 - 1.0)
        # reduction='none' so we can record per-model losses; sum across models
        # gives independent gradients per model (each model's params only see
        # its own loss slice).
        per_point = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss_per_model = per_point.mean(dim=1)  # (B,)
        loss = loss_per_model.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_vals = loss_per_model.detach().cpu().tolist()
        for i in range(B):
            losses_per_model[i].append(loss_vals[i])

        if step % 250 == 0 or step == steps - 1:
            with torch.no_grad():
                acc_per_model = ((torch.sigmoid(logits) > 0.5).float() == labels).float().mean(dim=1)
            mean_loss = loss_per_model.mean().item()
            mean_acc = acc_per_model.mean().item()
            print(f"step {step:5d}  mean_loss {mean_loss:.4f}  mean_acc {mean_acc:.3f}")
            if wandb_prefixes is not None:
                acc_vals = acc_per_model.cpu().tolist()
                log = {}
                for i, prefix in enumerate(wandb_prefixes):
                    log[f"{prefix}/loss"] = loss_vals[i]
                    log[f"{prefix}/acc"] = acc_vals[i]
                    log[f"{prefix}/step"] = step
                wandb.log(log)
    return model, losses_per_model


def extract_head_state_dict(batched_model, head_idx):
    """Convert one head of a BatchedPointClassifier into a state_dict that can
    be loaded directly into a fresh PointClassifier(hidden=batched_model.hidden).

    nn.Linear stores weights as (out, in); our batched weights are (B, in, out),
    so each head needs a transpose.
    """
    sd = batched_model.state_dict()
    return {
        "net.0.weight": sd["W1"][head_idx].t().contiguous().cpu(),
        "net.0.bias": sd["b1"][head_idx].contiguous().cpu(),
        "net.2.weight": sd["W2"][head_idx].t().contiguous().cpu(),
        "net.2.bias": sd["b2"][head_idx].contiguous().cpu(),
        "net.4.weight": sd["W3"][head_idx].t().contiguous().cpu(),
        "net.4.bias": sd["b3"][head_idx].contiguous().cpu(),
    }


def batched_evaluate_grid(model, size, device):
    """Evaluate a BatchedPointClassifier on a (size, size) grid for each head.
    Returns a (B, size, size) numpy array of probabilities."""
    B = model.n_models
    ys = torch.linspace(0.0, 1.0, size, device=device)
    xs = torch.linspace(0.0, 1.0, size, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (N, 2)
    xy = xy.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, N, 2)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(xy * 2.0 - 1.0))
    return probs.view(B, size, size).cpu().numpy()


def plot_svg_and_prediction(svg_img, probs, description, fig=None, axes=None):
    """Render a 2-panel figure: original SVG on the left, classifier P(inside)
    on the right with the p=0.5 decision boundary overlaid."""
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

    axes[0].imshow(svg_img, extent=[0.0, 1.0, 1.0, 0.0])
    title = description[:60] + ("..." if len(description) > 60 else "")
    axes[0].set_title(f"Original SVG\n{title}", fontsize=9)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")

    im = axes[1].imshow(
        probs,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=[0.0, 1.0, 1.0, 0.0],
    )
    xx = np.linspace(0.0, 1.0, probs.shape[1])
    yy = np.linspace(0.0, 1.0, probs.shape[0])
    axes[1].contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
    axes[1].set_title("Classifier P(inside)\nblack = decision boundary", fontsize=9)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    return fig, axes


def run_single_svg():
    """Train the classifier on a single SVG and save plots locally (legacy entry point)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    svg_xml, description = load_one_svg(skip=10)
    print(f"Loaded SVG. Description: {description[:100]}")

    mask = rasterize_svg_mask(svg_xml, size=4096)
    print(f"Mask shape: {mask.shape}  inside fraction: {mask.mean():.3f}")

    model = PointClassifier(hidden=16).to(device)
    model, losses = train(model, mask, steps=20000, batch_size=2**16, lr=1e-3, device=device)

    plot_resolution = 512
    probs = evaluate_grid(model, size=plot_resolution, device=device)
    svg_img = render_svg_to_image(svg_xml, plot_resolution)

    fig, _ = plot_svg_and_prediction(svg_img, probs, description)
    fig.tight_layout()
    out_path = "classifier_svg.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(losses, linewidth=0.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("step")
    ax2.set_ylabel("BCE loss (log)")
    ax2.set_title("Training loss")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    fig2.tight_layout()
    loss_path = "classifier_svg_loss.png"
    fig2.savefig(loss_path, dpi=150)
    print(f"Saved loss curve to {loss_path}")

    plt.show()


def main():
    """Train one tiny classifier per SVG, over the entire dataset, in chunks
    trained in parallel. Parameters are saved to classifier_parameters/ for
    every SVG; per-SVG visualization plots (and the combined grid / loss
    curves / accuracy bar chart) are only produced for the first PLOT_LIMIT
    SVGs, since the dataset is too large to reasonably visualize all of them."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(PARAMS_DIR, exist_ok=True)

    PLOT_LIMIT = 500
    chunk_size = 32
    steps = 20000
    batch_size = 2**16
    lr = 1e-3
    hidden = 16
    plot_resolution = 512
    mask_resolution = 4096

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading all SVGs from dataset (this may take a while)...")
    svgs = load_svgs(n=None, skip=10)
    n_svgs = len(svgs)
    n_plots = min(n_svgs, PLOT_LIMIT)
    print(f"Loaded {n_svgs} SVGs (plotting first {n_plots})")

    config = {
        "n_svgs": n_svgs,
        "plot_limit": PLOT_LIMIT,
        "chunk_size": chunk_size,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "hidden": hidden,
        "plot_resolution": plot_resolution,
        "mask_resolution": mask_resolution,
    }
    wandb.init(project="svg_classifier", config=config)

    # Custom step axis per SVG so each learning curve plots cleanly in wandb.
    # Only defined for the SVGs we actually stream per-step curves for.
    for i in range(n_plots):
        prefix = f"svg_{i:06d}"
        wandb.define_metric(f"{prefix}/step")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")

    final_accs = [None] * n_svgs
    inside_fractions = [None] * n_svgs
    all_losses = [None] * n_plots  # only kept for SVGs we plot
    cached_results = [None] * n_plots  # (svg_img, probs, description) for first n_plots

    for chunk_start in range(0, n_svgs, chunk_size):
        chunk = svgs[chunk_start:chunk_start + chunk_size]
        B = len(chunk)
        chunk_indices = list(range(chunk_start, chunk_start + B))
        prefixes = [f"svg_{i:06d}" for i in chunk_indices]
        print(f"\n=== Chunk {chunk_start // chunk_size + 1}: training {B} models in parallel "
              f"({prefixes[0]}..{prefixes[-1]}) ===")

        masks_np = []
        for j, (svg_xml, description) in enumerate(chunk):
            mask = rasterize_svg_mask(svg_xml, size=mask_resolution)
            inside_fraction = float(mask.mean())
            inside_fractions[chunk_indices[j]] = inside_fraction
            print(f"  {prefixes[j]}  inside_frac={inside_fraction:.3f}  {description[:70]}")
            masks_np.append(mask)
        masks_tensor = torch.from_numpy(np.stack(masks_np)).to(device)

        # Only stream per-step training curves to wandb for chunks that
        # contain SVGs within the plot limit.
        log_curves = chunk_start < n_plots

        model = BatchedPointClassifier(n_models=B, hidden=hidden).to(device)
        model, losses_per_model = batched_train(
            model, masks_tensor,
            steps=steps, batch_size=batch_size, lr=lr,
            wandb_prefixes=prefixes if log_curves else None,
        )

        # Final accuracies on a fresh random sample, computed in parallel.
        model.eval()
        with torch.no_grad():
            test_xy = torch.rand(B, 2**16, 2, device=device)
            test_labels = batched_mask_lookup(masks_tensor, test_xy)
            test_logits = model(test_xy * 2.0 - 1.0)
            test_acc_per_model = (
                (torch.sigmoid(test_logits) > 0.5).float() == test_labels
            ).float().mean(dim=1).cpu().tolist()

        # Only evaluate the visualization grid if any SVG in this chunk is
        # within the plot limit.
        probs_batch = None
        if log_curves:
            probs_batch = batched_evaluate_grid(model, size=plot_resolution, device=device)

        for j in range(B):
            global_idx = chunk_indices[j]
            prefix = prefixes[j]
            svg_xml, description = chunk[j]
            test_acc = test_acc_per_model[j]
            losses = losses_per_model[j]

            final_accs[global_idx] = test_acc
            print(f"  {prefix}  final_acc={test_acc:.4f}")

            # Save this head's weights as a loadable PointClassifier state dict
            # for EVERY SVG (not just the plotted ones).
            ckpt_path = os.path.join(PARAMS_DIR, f"{prefix}.pt")
            torch.save(
                {
                    "state_dict": extract_head_state_dict(model, j),
                    "hidden": hidden,
                    "description": description,
                    "final_acc": test_acc,
                    "inside_fraction": inside_fractions[global_idx],
                },
                ckpt_path,
            )

            # Per-SVG 2-panel plot + combined-grid cache only for the first
            # PLOT_LIMIT SVGs.
            if global_idx < n_plots:
                all_losses[global_idx] = losses
                probs = probs_batch[j]
                svg_img = render_svg_to_image(svg_xml, plot_resolution)
                cached_results[global_idx] = (svg_img, probs, description)

                fig, _ = plot_svg_and_prediction(svg_img, probs, description)
                fig.tight_layout()
                out_path = os.path.join(PLOTS_DIR, f"{prefix}.png")
                fig.savefig(out_path, dpi=150)
                print(f"  saved {out_path}")

                wandb.log({
                    f"{prefix}/result": wandb.Image(fig),
                    f"{prefix}/final_acc": test_acc,
                    f"{prefix}/inside_fraction": inside_fractions[global_idx],
                })
                plt.close(fig)

        # Free per-chunk GPU memory before the next chunk.
        del model, masks_tensor, test_xy, test_labels, test_logits
        if probs_batch is not None:
            del probs_batch
        if device == "cuda":
            torch.cuda.empty_cache()

    # Combined grid of (svg, prediction) pairs for the first n_plots SVGs.
    # `pairs_per_row` pairs per row, each pair occupying two adjacent cells.
    pairs_per_row = 4
    n_rows = (n_plots + pairs_per_row - 1) // pairs_per_row
    n_cols = pairs_per_row * 2
    grid_fig, grid_axes = plt.subplots(
        n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows)
    )
    if n_rows == 1:
        grid_axes = grid_axes[np.newaxis, :]
    for i, entry in enumerate(cached_results):
        if entry is None:
            continue
        svg_img, probs, description = entry
        row = i // pairs_per_row
        col_pair = (i % pairs_per_row) * 2
        ax_svg = grid_axes[row, col_pair]
        ax_pred = grid_axes[row, col_pair + 1]
        ax_svg.imshow(svg_img, extent=[0.0, 1.0, 1.0, 0.0])
        title = description[:40] + ("..." if len(description) > 40 else "")
        ax_svg.set_title(f"svg_{i:06d}\n{title}", fontsize=8)
        ax_svg.set_aspect("equal")
        ax_svg.set_xticks([])
        ax_svg.set_yticks([])
        ax_pred.imshow(
            probs, cmap="RdBu_r", vmin=0.0, vmax=1.0,
            origin="upper", extent=[0.0, 1.0, 1.0, 0.0],
        )
        xx = np.linspace(0.0, 1.0, probs.shape[1])
        yy = np.linspace(0.0, 1.0, probs.shape[0])
        ax_pred.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.0)
        ax_pred.set_title(f"acc={final_accs[i]:.3f}", fontsize=8)
        ax_pred.set_aspect("equal")
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
    grid_fig.tight_layout()
    grid_path = os.path.join(PLOTS_DIR, "all_results.png")
    grid_fig.savefig(grid_path, dpi=120)
    print(f"\nSaved combined grid to {grid_path}")
    wandb.log({"all_results": wandb.Image(grid_fig)})
    plt.close(grid_fig)

    # Combined loss curves (log scale) for the first n_plots SVGs. With
    # hundreds of curves a legend is useless, so we omit it beyond a threshold.
    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    for i, losses in enumerate(all_losses):
        if losses is None:
            continue
        loss_ax.plot(losses, linewidth=0.4, alpha=0.5)
    loss_ax.set_yscale("log")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("BCE loss (log)")
    loss_ax.set_title(f"Training loss curves (first {n_plots} SVGs)")
    loss_ax.grid(True, which="both", linestyle=":", alpha=0.5)
    loss_fig.tight_layout()
    loss_path = os.path.join(PLOTS_DIR, "loss_curves.png")
    loss_fig.savefig(loss_path, dpi=150)
    print(f"Saved loss curves to {loss_path}")
    wandb.log({"loss_curves": wandb.Image(loss_fig)})
    plt.close(loss_fig)

    # Final accuracy bar chart for the first n_plots SVGs.
    plotted_accs = final_accs[:n_plots]
    acc_fig, acc_ax = plt.subplots(figsize=(min(40, max(8, 0.03 * n_plots)), 4))
    acc_ax.bar(range(n_plots), plotted_accs, color="steelblue", width=1.0)
    acc_ax.set_xlim(-0.5, n_plots - 0.5)
    acc_ax.set_xlabel("SVG index")
    acc_ax.set_ylabel("final accuracy")
    acc_ax.set_ylim(0.0, 1.0)
    mean_plotted = float(np.mean(plotted_accs))
    acc_ax.axhline(mean_plotted, color="black", linestyle="--",
                   label=f"mean={mean_plotted:.3f}")
    acc_ax.set_title(f"Final accuracy per SVG (first {n_plots})")
    acc_ax.legend()
    acc_ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    acc_fig.tight_layout()
    acc_path = os.path.join(PLOTS_DIR, "final_accuracies.png")
    acc_fig.savefig(acc_path, dpi=150)
    print(f"Saved accuracy bar chart to {acc_path}")
    wandb.log({
        "final_accuracies": wandb.Image(acc_fig),
        "mean_final_acc": float(np.mean(final_accs)),
        "min_final_acc": float(np.min(final_accs)),
        "max_final_acc": float(np.max(final_accs)),
    })
    plt.close(acc_fig)

    # Summary table of per-SVG metrics (covers all SVGs, not just plotted).
    table = wandb.Table(columns=["svg_idx", "description", "inside_fraction", "final_acc"])
    for i, (_, description) in enumerate(svgs):
        table.add_data(i, description[:120], inside_fractions[i], final_accs[i])
    wandb.log({"summary": table})

    wandb.finish()
    print(f"\nDone. mean final acc = {np.mean(final_accs):.4f}")


if __name__ == "__main__":
    main()
