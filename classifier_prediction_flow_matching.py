"""Flow matching over classifier parameters.

We treat each saved point-classifier (a tiny MLP with 337 parameters that
learned to label points inside/outside a single SVG) as one data point in a
337-dimensional space, and train a neural velocity field that pushes samples
from N(0, I) to the distribution of these parameter vectors under the
conditional OT flow-matching objective.

Per-classifier parameter layout (PointClassifier(hidden=16)):

    net.0.weight  (16, 2)    32
    net.0.bias    (16,)      16
    net.2.weight  (16,16)   256
    net.2.bias    (16,)      16
    net.4.weight  (1, 16)    16
    net.4.bias    (1,)        1
    ----------------------------
    total                   337

Training data lives under classifier_parameters/ (24 files). The meta-model
is a ResNet-style MLP on the flattened, dataset-normalized parameter vector
with a sinusoidal time embedding. Periodically we sample a few classifiers
from noise, materialize them into fresh PointClassifier modules, evaluate
them on a grid, and log the resulting probability maps to wandb.
"""

import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from classifier_svg import PointClassifier, evaluate_grid

PARAMS_DIR = "classifier_parameters"
PLOTS_DIR = "flow_matching_plots"

# Fixed param layout used everywhere in this file.
PARAM_SHAPES = [
    ("net.0.weight", (16, 2)),
    ("net.0.bias", (16,)),
    ("net.2.weight", (16, 16)),
    ("net.2.bias", (16,)),
    ("net.4.weight", (1, 16)),
    ("net.4.bias", (1,)),
]
TOTAL_PARAMS = sum(int(np.prod(s)) for _, s in PARAM_SHAPES)  # 337
HIDDEN = 16


# ---------------------------------------------------------------------------
# Data loading / (de)flattening
# ---------------------------------------------------------------------------

def flatten_state_dict(sd):
    """Flatten a PointClassifier state_dict into a (TOTAL_PARAMS,) tensor."""
    parts = []
    for name, shape in PARAM_SHAPES:
        t = sd[name]
        assert tuple(t.shape) == shape, f"{name}: got {tuple(t.shape)}, expected {shape}"
        parts.append(t.reshape(-1))
    return torch.cat(parts, dim=0)


def unflatten_to_state_dict(vec):
    """Inverse of flatten_state_dict. vec: (TOTAL_PARAMS,) tensor."""
    assert vec.numel() == TOTAL_PARAMS
    sd = {}
    offset = 0
    for name, shape in PARAM_SHAPES:
        n = int(np.prod(shape))
        sd[name] = vec[offset:offset + n].reshape(*shape).contiguous()
        offset += n
    return sd


def load_classifier_dataset(params_dir=PARAMS_DIR):
    """Load all classifier parameter files and stack into a (N, TOTAL_PARAMS) tensor.

    Returns the flat tensor and a list of metadata dicts (description, etc.).
    """
    paths = sorted(glob.glob(os.path.join(params_dir, "*.pt")))
    assert len(paths) > 0, f"no classifier params found in {params_dir}"
    vecs = []
    metas = []
    for p in paths:
        blob = torch.load(p, map_location="cpu", weights_only=False)
        assert blob["hidden"] == HIDDEN, f"{p}: hidden={blob['hidden']} != {HIDDEN}"
        vecs.append(flatten_state_dict(blob["state_dict"]))
        metas.append({
            "path": p,
            "name": os.path.splitext(os.path.basename(p))[0],
            "description": blob.get("description", ""),
            "final_acc": float(blob.get("final_acc", float("nan"))),
        })
    data = torch.stack(vecs, dim=0)  # (N, P)
    return data, metas


# ---------------------------------------------------------------------------
# Flow matching velocity network
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Classic sinusoidal embedding for a scalar t in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "time embedding dim must be even"
        self.dim = dim
        half = dim // 2
        # log-spaced frequencies similar to the Transformer positional encoding.
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t):
        # t: (B,) in [0, 1]
        args = t.unsqueeze(-1) * self.freqs.unsqueeze(0) * (2.0 * math.pi)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block with FiLM conditioning on the time embedding."""

    def __init__(self, dim, t_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        # FiLM: produce (scale, shift) from time embedding.
        self.film = nn.Linear(t_dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, t_emb):
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        y = self.norm(h)
        y = y * (1.0 + scale) + shift
        y = F.silu(self.lin1(y))
        y = self.dropout(y)
        y = self.lin2(y)
        return h + y


class FlowMatchingModel(nn.Module):
    """Velocity field v_theta(x_t, t) for x in R^TOTAL_PARAMS.

    Design notes:
    - Input dimension is tiny (337), so we project it up to a wider hidden
      space before running residual blocks.
    - Time is injected with FiLM inside every block so gradients w.r.t. t
      reach every layer directly.
    - Output is the per-coordinate velocity, same shape as input.
    """

    def __init__(self, param_dim=TOTAL_PARAMS, hidden=512, n_blocks=6,
                 t_dim=128, dropout=0.0):
        super().__init__()
        self.param_dim = param_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_proj = nn.Linear(param_dim, hidden)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, t_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, param_dim)
        # Zero-init the output so the initial velocity field is 0 (the typical
        # flow-matching / diffusion initialization trick: training starts from
        # an identity-mapping flow).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, t):
        # x: (B, P), t: (B,) in [0, 1]
        t_emb = self.time_embed(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.out_norm(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Flow matching loss and sampling
# ---------------------------------------------------------------------------

def flow_matching_loss(model, x1, eps=1e-4):
    """Conditional OT flow matching loss.

    x1: (B, P) normalized data samples.
    - x0 ~ N(0, I)
    - x_t = (1 - t) x0 + t x1
    - target velocity = x1 - x0
    Loss: MSE between model(x_t, t) and the target velocity.
    """
    B = x1.shape[0]
    device = x1.device
    x0 = torch.randn_like(x1)
    # Sample t away from the exact endpoints for a little numerical slack.
    t = torch.rand(B, device=device) * (1.0 - 2.0 * eps) + eps
    t_broadcast = t.view(B, 1)
    x_t = (1.0 - t_broadcast) * x0 + t_broadcast * x1
    v_target = x1 - x0
    v_pred = model(x_t, t)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample(model, n_samples, n_steps, device, param_dim=TOTAL_PARAMS):
    """Integrate the learned velocity field from N(0, I) at t=0 to t=1
    using a fixed-step Euler scheme. Returns (n_samples, P) tensor."""
    model.eval()
    x = torch.randn(n_samples, param_dim, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=device)
        v = model(x, t)
        x = x + dt * v
    return x


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def materialize_and_evaluate(normed_params, mean, std, grid_size=128, device="cpu"):
    """Denormalize a batch of flat parameter vectors, load each into a
    PointClassifier, and evaluate on a (grid_size, grid_size) probability grid.

    normed_params: (B, P) tensor in the normalized space.
    Returns a (B, grid_size, grid_size) numpy array of P(inside).
    """
    params = normed_params * std + mean  # denormalize
    probs_list = []
    for i in range(params.shape[0]):
        sd = unflatten_to_state_dict(params[i].cpu())
        clf = PointClassifier(hidden=HIDDEN).to(device)
        clf.load_state_dict(sd)
        probs = evaluate_grid(clf, size=grid_size, device=device)
        probs_list.append(probs)
    return np.stack(probs_list, axis=0)


def plot_sampled_classifiers(probs_batch, step, data_probs_batch=None):
    """Make a figure with one subplot per sampled classifier (and optionally
    a row of data examples for reference). Returns a matplotlib Figure."""
    n_sampled = probs_batch.shape[0]
    has_data = data_probs_batch is not None
    n_rows = 2 if has_data else 1
    n_cols = max(n_sampled, data_probs_batch.shape[0] if has_data else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows),
                              squeeze=False)

    def show(ax, probs, title):
        ax.imshow(probs, cmap="RdBu_r", vmin=0.0, vmax=1.0,
                  origin="upper", extent=[0.0, 1.0, 1.0, 0.0])
        xx = np.linspace(0.0, 1.0, probs.shape[1])
        yy = np.linspace(0.0, 1.0, probs.shape[0])
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    sample_row = 0 if not has_data else 1
    for j in range(n_cols):
        if j < n_sampled:
            show(axes[sample_row, j], probs_batch[j], f"sample {j}")
        else:
            axes[sample_row, j].axis("off")
    if has_data:
        for j in range(n_cols):
            if j < data_probs_batch.shape[0]:
                show(axes[0, j], data_probs_batch[j], f"data {j}")
            else:
                axes[0, j].axis("off")

    suptitle = f"Flow-matching samples @ step {step}"
    if has_data:
        suptitle += "  (top: training examples, bottom: samples)"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_flow_matching(
    epochs=2000,
    batch_size=32,
    lr=2e-4,
    weight_decay=1e-5,
    hidden=512,
    n_blocks=6,
    t_dim=128,
    dropout=0.0,
    ema_decay=0.999,
    sample_every=100,
    log_every=10,
    n_sample_steps=100,
    n_samples_to_plot=6,
    grid_size=128,
    seed=0,
    device=None,
    wandb_project="svg_classifier_flow_matching",
    resume=None,
    start_epoch=0,
):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # When resuming, perturb the seed so we don't replay the exact same noise.
    effective_seed = seed + start_epoch
    torch.manual_seed(effective_seed)
    np.random.seed(effective_seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----- data -----
    data, metas = load_classifier_dataset()
    N, P = data.shape
    print(f"Loaded {N} classifiers with {P} params each")
    assert P == TOTAL_PARAMS

    # Train/test split (80/20, at least 1 test sample).
    perm = torch.randperm(N)
    n_test = max(1, int(0.2 * N))
    n_train = N - n_test
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    train_data = data[train_idx]
    test_data = data[test_idx]
    print(f"Train/test split: {n_train} train, {n_test} test")

    # Per-dim normalization computed on training set only.
    mean = train_data.mean(dim=0, keepdim=True)
    std = train_data.std(dim=0, keepdim=True).clamp(min=1e-4)
    train_n = ((train_data - mean) / std).to(device)
    test_n = ((test_data - mean) / std).to(device)
    mean_dev = mean.to(device)
    std_dev = std.to(device)

    # Precompute a reference figure of a few training examples to anchor the
    # comparison in the periodic sample plots.
    ref_idx = list(range(min(n_samples_to_plot, n_train)))
    data_probs_ref = materialize_and_evaluate(
        train_n[ref_idx], mean_dev, std_dev, grid_size=grid_size, device=device,
    )

    # ----- model -----
    model = FlowMatchingModel(
        param_dim=P, hidden=hidden, n_blocks=n_blocks,
        t_dim=t_dim, dropout=dropout,
    ).to(device)
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"Flow model: {n_model_params/1e6:.3f}M parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Simple EMA of parameters for cleaner samples.
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if resume is not None:
        print(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        ema = {k: v.to(device) for k, v in ckpt["ema_state_dict"].items()}
        print(f"Loaded model + EMA from epoch {start_epoch}")

    def ema_update():
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)
                else:
                    ema[k].copy_(v)

    def with_ema_weights(fn):
        """Temporarily swap model state for EMA state and run fn()."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        try:
            return fn()
        finally:
            model.load_state_dict(backup)

    # ----- wandb -----
    wandb.init(
        project=wandb_project,
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "hidden": hidden,
            "n_blocks": n_blocks,
            "t_dim": t_dim,
            "dropout": dropout,
            "ema_decay": ema_decay,
            "n_sample_steps": n_sample_steps,
            "grid_size": grid_size,
            "n_data": N,
            "n_train": n_train,
            "n_test": n_test,
            "param_dim": P,
            "model_params": n_model_params,
        },
    )

    # ----- training loop -----
    n_test_evals = 8  # number of MC evaluations to average for test loss
    global_step = 0
    model.train()
    final_epoch = start_epoch + epochs
    for epoch in range(start_epoch + 1, final_epoch + 1):
        # Shuffle training data each epoch and iterate in batches.
        perm_epoch = torch.randperm(n_train, device=device)
        epoch_losses = []
        for i in range(0, n_train, batch_size):
            idx = perm_epoch[i:i + batch_size]
            x1 = train_n[idx]
            loss = flow_matching_loss(model, x1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            ema_update()
            global_step += 1
            epoch_losses.append(loss.item())

        train_loss_avg = sum(epoch_losses) / len(epoch_losses)

        if epoch % log_every == 0 or epoch == start_epoch + 1:
            # Compute test loss (averaged over a few MC draws for stability).
            model.eval()
            with torch.no_grad():
                test_losses = []
                for _ in range(n_test_evals):
                    test_loss = flow_matching_loss(model, test_n)
                    test_losses.append(test_loss.item())
                test_loss_avg = sum(test_losses) / len(test_losses)
            model.train()
            wandb.log({
                "train/loss": train_loss_avg,
                "test/loss": test_loss_avg,
                "epoch": epoch,
            }, step=global_step)
            print(f"epoch {epoch:5d}  train {train_loss_avg:.5f}  test {test_loss_avg:.5f}")

        if epoch % sample_every == 0 or epoch == final_epoch:
            def _sample_and_plot():
                samples = sample(
                    model, n_samples_to_plot, n_sample_steps, device, param_dim=P,
                )
                probs_batch = materialize_and_evaluate(
                    samples, mean_dev, std_dev, grid_size=grid_size, device=device,
                )
                return samples, probs_batch

            samples, probs_batch = with_ema_weights(_sample_and_plot)

            fig = plot_sampled_classifiers(
                probs_batch, step=epoch, data_probs_batch=data_probs_ref,
            )
            out_path = os.path.join(PLOTS_DIR, f"samples_epoch_{epoch:07d}.png")
            fig.savefig(out_path, dpi=120)
            print(f"  saved {out_path}")
            wandb.log({"samples/figure": wandb.Image(fig), "epoch": epoch},
                      step=global_step)
            plt.close(fig)

            # Diagnostics: how far the sampled params drift from the data in
            # the normalized space. Collapsing to a single point is the usual
            # failure mode with N=24.
            with torch.no_grad():
                s_mean = samples.mean().item()
                s_std = samples.std().item()
                # Mean pairwise distance between samples.
                if samples.shape[0] > 1:
                    diffs = samples.unsqueeze(0) - samples.unsqueeze(1)
                    pair_dist = diffs.norm(dim=-1)
                    mask = ~torch.eye(
                        samples.shape[0], dtype=torch.bool, device=device
                    )
                    mean_pair_dist = pair_dist[mask].mean().item()
                else:
                    mean_pair_dist = 0.0
            wandb.log({
                "samples/mean": s_mean,
                "samples/std": s_std,
                "samples/mean_pair_dist": mean_pair_dist,
                "epoch": epoch,
            }, step=global_step)

            model.train()

    # Save a final checkpoint (EMA weights) plus the normalization stats so
    # the model can be resampled later.
    ckpt_path = os.path.join(PLOTS_DIR, "flow_model.pt")
    torch.save(
        {
            "ema_state_dict": ema,
            "model_state_dict": model.state_dict(),
            "mean": mean.cpu(),
            "std": std.cpu(),
            "config": {
                "hidden": hidden,
                "n_blocks": n_blocks,
                "t_dim": t_dim,
                "dropout": dropout,
                "param_dim": P,
            },
        },
        ckpt_path,
    )
    print(f"Saved flow model checkpoint to {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a flow_model.pt checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of epochs to train (additional epochs when resuming)")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="Epoch number the resumed checkpoint left off at")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_flow_matching(
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        start_epoch=args.start_epoch,
    )
