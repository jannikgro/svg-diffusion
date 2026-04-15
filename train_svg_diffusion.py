import argparse
import io
import os
import random

import torch
import wandb
import time
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import cairosvg

from prepare_dataset import setup_tokenizer, create_dataloader
from model import SVGDiffusionModel
from svg_utils import decode_to_svg, reconstruct_svg

LIVE_CONFIG_PATH = "live_config.yaml"


def load_live_config():
    """Load the live configuration file, returning overrides or empty dict."""
    if os.path.exists(LIVE_CONFIG_PATH):
        try:
            with open(LIVE_CONFIG_PATH) as f:
                cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
        except Exception as e:
            print(f"  [live_config] Failed to load: {e}")
    return {}


def apply_live_config(optimizer, config, global_step):
    """Reload live_config.yaml and apply supported hyperparameter changes."""
    print(f"  [live_config] Step {global_step}: reloading {LIVE_CONFIG_PATH}")
    live = load_live_config()
    if not live:
        return

    changed = []
    if "lr" in live and live["lr"] != config.get("_live_lr"):
        for pg in optimizer.param_groups:
            pg["lr"] = live["lr"]
        config["_live_lr"] = live["lr"]
        changed.append(f"lr={live['lr']}")

    if "grad_clip" in live and live["grad_clip"] != config.get("grad_clip"):
        config["grad_clip"] = live["grad_clip"]
        changed.append(f"grad_clip={live['grad_clip']}")

    if "eval_every" in live and live["eval_every"] != config.get("eval_every"):
        config["eval_every"] = live["eval_every"]
        changed.append(f"eval_every={live['eval_every']}")

    if "ema_decay" in live and live["ema_decay"] != config.get("ema_decay"):
        config["ema_decay"] = live["ema_decay"]
        changed.append(f"ema_decay={live['ema_decay']}")

    if "checkpoint_every" in live and live["checkpoint_every"] != config.get("checkpoint_every"):
        config["checkpoint_every"] = live["checkpoint_every"]
        changed.append(f"checkpoint_every={live['checkpoint_every']}")

    if "skip_val" in live and live["skip_val"] != config.get("skip_val"):
        config["skip_val"] = live["skip_val"]
        changed.append(f"skip_val={live['skip_val']}")

    if changed:
        print(f"  [live_config] Step {global_step}: updated {', '.join(changed)}")


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Swap model params with EMA params; call restore() to undo."""
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        """Restore original model params after apply()."""
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def flow_matching_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    segment_ids = batch["segment_ids"].to(device)
    coord_mask = batch["coord_mask"].to(device)
    coord_values = batch["coord_values"].to(device)

    B = input_ids.shape[0]
    t = torch.rand(B, device=device)
    noise = torch.randn_like(coord_values) * coord_mask.float()
    noisy = (1 - t.unsqueeze(1)) * noise + t.unsqueeze(1) * coord_values
    target_v = coord_values - noise

    pred_v = model(input_ids, attention_mask, segment_ids, noisy, coord_mask, t)
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
def reconstruct_samples(model, val_loader, device, step, num_samples=10,
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
                "segment_ids": batch["segment_ids"][i],
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
        seg = item["segment_ids"].unsqueeze(0).to(device)
        cmask = item["coord_mask"].unsqueeze(0).to(device)
        pred_all = model.sample(ids, attn, seg, cmask, num_steps=50)
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

    plt.suptitle(f"Step {step}", fontsize=12)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step:06d}.png")
    fig.savefig(path, dpi=150)
    wandb.log({"val/reconstructions": wandb.Image(fig), "step": step})
    plt.close(fig)
    print(f"  Saved reconstructions to {path}")

    model.train()


def save_checkpoint(model, optimizer, scheduler, ema, global_step, epoch, val_loss, config, path):
    """Save a training checkpoint atomically to avoid corruption on crash."""
    tmp_path = path + ".tmp"
    torch.save({
        "global_step": global_step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "ema_shadow": ema.shadow,
        "val_loss": val_loss,
        "config": config,
    }, tmp_path)
    os.replace(tmp_path, path)


def train(model, train_loader, val_loader, optimizer, scheduler, ema, device,
          config, epochs=10, checkpoint_dir="checkpoints", reconstruction_dir="reconstructions",
          grad_clip=1.0, accumulation_steps=1, eval_every=500,
          start_step=0, start_epoch=0):
    model.train()
    global_step = start_step
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def _eval_and_checkpoint(epoch):
        nonlocal best_val_loss
        skip_val = config.get("skip_val", False)

        val_loss = None
        if not skip_val:
            ema.apply(model)
            val_loss = evaluate(model, val_loader, device)
            wandb.log({"val/loss": val_loss, "step": global_step})
            print(f"  Step {global_step} | Val: {val_loss:.6f} | Train loss: {last_loss:.4f} | "
                  f"Grad norm: {last_grad_norm:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Speed: {last_speed:.2f} it/s")
            reconstruct_samples(model, val_loader, device, global_step,
                                out_dir=reconstruction_dir)
            ema.restore(model)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, ema, global_step, epoch, val_loss, config,
                                os.path.join(checkpoint_dir, "best.pt"))
                print(f"  New best val loss: {val_loss:.6f}")
        else:
            print(f"  Step {global_step} | Train loss: {last_loss:.4f} | "
                  f"Grad norm: {last_grad_norm:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Speed: {last_speed:.2f} it/s (val skipped)")

        checkpoint_every = config.get("checkpoint_every", config["eval_every"])
        if global_step % checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, ema, global_step, epoch,
                            val_loss if val_loss is not None else -1, config,
                            os.path.join(checkpoint_dir, "latest.pt"))
        model.train()

    last_loss, last_grad_norm, last_speed = 0.0, 0.0, 0.0
    step_t0 = time.monotonic()

    for epoch in range(start_epoch, epochs):
        total_loss, count = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            loss = flow_matching_loss(model, batch, device)
            (loss / accumulation_steps).backward()
            total_loss += loss.item()
            count += 1

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
                optimizer.step()
                scheduler.step()
                ema.update(model)
                optimizer.zero_grad()
                global_step += 1

                now = time.monotonic()
                last_speed = 1.0 / max(now - step_t0, 1e-9)
                step_t0 = now
                last_loss = loss.item()
                last_grad_norm = grad_norm.item()

                # Reload live config every 100 steps
                if global_step % 100 == 0:
                    apply_live_config(optimizer, config, global_step)
                    ema.decay = config.get("ema_decay", ema.decay)

                clipped = 1.0 if last_grad_norm > config["grad_clip"] else 0.0
                wandb.log({
                    "train/step_loss": last_loss,
                    "train/grad_norm": last_grad_norm,
                    "train/grad_clipped": clipped,
                    "train/speed": last_speed,
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                })

                if global_step % config["eval_every"] == 0:
                    _eval_and_checkpoint(epoch)

        train_loss = total_loss / max(count, 1)
        wandb.log({"train/epoch_loss": train_loss, "epoch": epoch + 1})
        print(f"Epoch {epoch + 1}/{epochs} | Train: {train_loss:.6f}")

        # Also eval at end of epoch if not just evaluated
        if global_step % config["eval_every"] != 0:
            _eval_and_checkpoint(epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory or .pt file to resume from")
    parser.add_argument("--overfit", action="store_true",
                        help="Overfit mode: train on 4x effective batch size, eval every 1000 steps")
    args = parser.parse_args()

    run_id = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    checkpoint_dir = f"checkpoints/{run_id}"
    reconstruction_dir = f"reconstructions/{run_id}"

    config = {
        "model_name": "microsoft/Phi-4-mini-instruct",
        "max_samples": None,
        "accumulation_steps": 32,
        "lr": 2e-4,
        "epochs": 1000,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_targets": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "weight_decay": 0.01,
        "val_split": 0.05,
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "warmup_epochs": 0.5,
        "eval_every": 500,
        "checkpoint_every": 500,
        "skip_val": True,
        "max_token_len": 2048,
    }

    # Resolve resume checkpoint path
    resume_path = None
    if args.resume:
        if os.path.isdir(args.resume):
            resume_path = os.path.join(args.resume, "latest.pt")
        else:
            resume_path = args.resume
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"Resuming from {resume_path}")

    print("[startup] Initializing wandb...")
    wandb.init(project="svg-diffusion", config=config,
               resume="allow" if resume_path else None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[startup] Device: {device}")
    print("[startup] Setting up tokenizer...")
    tokenizer = setup_tokenizer(config["model_name"])

    print("[startup] Loading dataset...")
    # Load dataset with no sequence-length truncation, skip overly long SVGs
    full_loader = create_dataloader(
        tokenizer,
        max_samples=config["max_samples"],
        batch_size=1,
        max_seq_len=1_000_000,
        max_prompt_len=1_000_000,
        max_token_len=config["max_token_len"],
    )

    dataset = full_loader.dataset

    if args.overfit:
        overfit_size = 4 * config["accumulation_steps"]
        overfit_size = min(overfit_size, len(dataset))
        train_ds, _ = torch.utils.data.random_split(
            dataset, [overfit_size, len(dataset) - overfit_size])
        val_ds = train_ds  # validate on the same data to check memorisation
        train_size, val_size = overfit_size, overfit_size
        config["eval_every"] = 1000
        config["checkpoint_every"] = 1000
        print(f"[overfit] Using {overfit_size} samples (4x effective batch size)")
    else:
        val_size = max(1, int(len(dataset) * config["val_split"]))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    from prepare_dataset import collate_single
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    print(f"Train: {train_size} samples ({len(train_loader)} batches) | Val: {val_size} samples ({len(val_loader)} batches)")

    print("[startup] Creating model...")
    model = SVGDiffusionModel(config["model_name"], tokenizer, lora_r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_targets=config["lora_targets"]).to(device)
    print("[startup] Creating optimizer...")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"], weight_decay=config["weight_decay"])

    # Cosine annealing with linear warmup (counted in optimizer steps, not samples)
    optimizer_steps_per_epoch = (len(train_loader) + config["accumulation_steps"] - 1) // config["accumulation_steps"]
    total_steps = optimizer_steps_per_epoch * config["epochs"]
    warmup_steps = optimizer_steps_per_epoch * config["warmup_epochs"]

    start_step = 0
    start_epoch = 0

    if resume_path:
        # No warmup when resuming — just cosine from current position
        print("[startup] Creating scheduler (cosine, no warmup)...")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        print(f"[startup] Loading checkpoint from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        print("[startup] Loading model state dict...")
        model.load_state_dict(ckpt["model_state_dict"])
        print("[startup] Loading optimizer state dict...")
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Don't load old scheduler state — we're using a fresh cosine-only scheduler
        if "ema_shadow" in ckpt:
            ema = EMA(model, decay=config["ema_decay"])
            ema.shadow = ckpt["ema_shadow"]
        else:
            ema = EMA(model, decay=config["ema_decay"])
        start_step = ckpt.get("global_step", 0)
        # Old checkpoints stored global_step in the "epoch" field, so don't
        # trust it — always restart epoch counting from 0 and rely on
        # global_step for progress tracking.
        start_epoch = 0
        print(f"Resumed: step={start_step}, val_loss={ckpt.get('val_loss', 'N/A')}")
    else:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=warmup_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
        ema = EMA(model, decay=config["ema_decay"])

    # Write initial live config file if it doesn't exist
    if not os.path.exists(LIVE_CONFIG_PATH):
        with open(LIVE_CONFIG_PATH, "w") as f:
            yaml.dump({
                "lr": config["lr"],
                "grad_clip": config["grad_clip"],
                "eval_every": config["eval_every"],
                "checkpoint_every": config["checkpoint_every"],
                "skip_val": config["skip_val"],
                "ema_decay": config["ema_decay"],
            }, f, default_flow_style=False)
        print(f"Created {LIVE_CONFIG_PATH} — edit to adjust hyperparameters on the fly")

    print("[startup] Starting training loop...")
    wandb.watch(model, log="gradients", log_freq=50)
    train(model, train_loader, val_loader, optimizer, scheduler, ema, device,
          config=config, epochs=config["epochs"], checkpoint_dir=checkpoint_dir,
          reconstruction_dir=reconstruction_dir, grad_clip=config["grad_clip"],
          accumulation_steps=config["accumulation_steps"],
          eval_every=config["eval_every"],
          start_step=start_step, start_epoch=start_epoch)
    wandb.finish()


if __name__ == "__main__":
    main()
