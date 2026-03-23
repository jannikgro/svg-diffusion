import io
import os
import random

import torch
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import cairosvg

from prepare_dataset import setup_tokenizer, create_dataloader
from model import SVGDiffusionModel
from svg_utils import decode_to_svg, reconstruct_svg


def flow_matching_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    coord_mask = batch["coord_mask"].to(device)
    coord_values = batch["coord_values"].to(device)

    B = input_ids.shape[0]
    t = torch.rand(B, device=device)
    noise = torch.randn_like(coord_values) * coord_mask.float()
    noisy = (1 - t.unsqueeze(1)) * noise + t.unsqueeze(1) * coord_values
    target_v = coord_values - noise

    pred_v = model(input_ids, attention_mask, noisy, coord_mask, t)
    return ((pred_v - target_v) ** 2 * coord_mask.float()).sum() / coord_mask.float().sum()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, count = 0.0, 0
    for batch in dataloader:
        loss = flow_matching_loss(model, batch, device)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def _render_svg_to_image(svg_xml, size=256):
    """Render an SVG XML string to a PIL Image, returning None on failure."""
    try:
        png = cairosvg.svg2png(bytestring=svg_xml.encode("utf-8"),
                               output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGB")
    except Exception as e:
        print(f"  [render error] {e}")
        print(f"  SVG snippet: {svg_xml[:200]}")
        return Image.new("RGB", (size, size), (200, 200, 200))


@torch.no_grad()
def reconstruct_samples(model, val_loader, device, epoch, num_samples=10,
                        out_dir="reconstructions"):
    """Run model sampling on random validation samples and save a side-by-side
    comparison of ground-truth vs reconstructed SVGs."""
    model.eval()

    # Collect all validation items from the loader
    all_items = []
    for batch in val_loader:
        B = batch["input_ids"].shape[0]
        for i in range(B):
            all_items.append({
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i],
                "coord_mask": batch["coord_mask"][i],
                "coord_values": batch["coord_values"][i],
                "coord_offset": batch["coord_offset"][i].item(),
                "coord_scale": batch["coord_scale"][i].item(),
                "skeleton": batch["skeleton"][i],
                "description": batch["description"][i],
            })

    selected = random.sample(all_items, min(num_samples, len(all_items)))

    fig, axes = plt.subplots(len(selected), 2, figsize=(6, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, 2)

    for row, item in enumerate(selected):
        skeleton = item["skeleton"]
        offset = item["coord_offset"]
        scale = item["coord_scale"]
        description = item["description"]

        # Ground truth: extract normalised coords at mask positions, denormalise
        mask = item["coord_mask"]
        gt_norm = item["coord_values"][mask].tolist()
        gt_svg_xml = decode_to_svg(reconstruct_svg(skeleton, gt_norm, offset, scale))

        # Model reconstruction: sample, then extract at mask positions
        ids = item["input_ids"].unsqueeze(0).to(device)
        attn = item["attention_mask"].unsqueeze(0).to(device)
        cmask = item["coord_mask"].unsqueeze(0).to(device)
        pred_all = model.sample(ids, attn, cmask, num_steps=50)
        pred_norm = pred_all[0][item["coord_mask"]].cpu().tolist()
        pred_svg_xml = decode_to_svg(reconstruct_svg(skeleton, pred_norm, offset, scale))

        gt_img = _render_svg_to_image(gt_svg_xml)
        pred_img = _render_svg_to_image(pred_svg_xml)

        prompt_label = description[:50] if description else f"Sample {row + 1}"
        axes[row, 0].imshow(gt_img)
        axes[row, 0].set_title(f"GT: {prompt_label}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_img)
        axes[row, 1].set_title(f"Pred: {prompt_label}", fontsize=8)
        axes[row, 1].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=12)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=150)
    wandb.log({"val/reconstructions": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
    print(f"  Saved reconstructions to {path}")

    model.train()


def train(model, train_loader, val_loader, optimizer, device, epochs=10):
    model.train()
    global_step = 0
    for epoch in range(epochs):
        total_loss, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            loss = flow_matching_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
            global_step += 1
            wandb.log({"train/loss": loss.item(), "step": global_step})
        train_loss = total_loss / max(count, 1)
        val_loss = evaluate(model, val_loader, device)
        wandb.log({"train/epoch_loss": train_loss, "val/loss": val_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch + 1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        reconstruct_samples(model, val_loader, device, epoch + 1)


def main():
    config = {
        "model_name": "microsoft/markuplm-base",
        "max_samples": 25000,
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 1000,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_seq_len": 512,
        "val_split": 0.05,
    }

    wandb.init(project="svg-diffusion", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = setup_tokenizer(config["model_name"])
    full_loader = create_dataloader(tokenizer, max_samples=config["max_samples"], batch_size=config["batch_size"], max_seq_len=config["max_seq_len"])

    dataset = full_loader.dataset
    val_size = max(1, int(len(dataset) * config["val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    from prepare_dataset import make_collate_fn
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    print(f"Train: {train_size} samples ({len(train_loader)} batches) | Val: {val_size} samples ({len(val_loader)} batches)")

    model = SVGDiffusionModel(config["model_name"], tokenizer, lora_r=config["lora_r"], lora_alpha=config["lora_alpha"]).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"])
    wandb.watch(model, log="gradients", log_freq=50)
    train(model, train_loader, val_loader, optimizer, device, epochs=config["epochs"])
    wandb.finish()


if __name__ == "__main__":
    main()
