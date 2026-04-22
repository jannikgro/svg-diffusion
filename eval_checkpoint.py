"""Load a training checkpoint and render a side-by-side GT vs predicted SVG grid.

Usage:
    python eval_checkpoint.py --checkpoint path/to/latest.pt [--num-samples 10]
                              [--num-val 50] [--num-steps 50] [--out eval.png]
"""
import argparse
import io
import os
import random

import cairosvg
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from model import SVGDiffusionModel
from prepare_dataset import collate_single, create_dataloader, setup_tokenizer
from svg_utils import decode_to_svg, reconstruct_svg


def _render_svg_to_image(svg_xml, size=256):
    try:
        png = cairosvg.svg2png(
            bytestring=svg_xml.encode("utf-8"),
            output_width=size, output_height=size,
        )
        return Image.open(io.BytesIO(png)).convert("RGB")
    except Exception as e:
        print(f"  [render error] {e}")
        return Image.new("RGB", (size, size), (200, 200, 200))


@torch.no_grad()
def reconstruct_and_save(model, val_loader, device, out_path,
                         num_samples=10, num_steps=50):
    model.eval()

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

    if not all_items:
        raise RuntimeError("No validation items available to sample from")

    selected = random.sample(all_items, min(num_samples, len(all_items)))
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, item in enumerate(selected):
        skeleton = item["skeleton"]
        offset = item["coord_offset"]
        scale = item["coord_scale"]
        description = item["description"]

        gt_norm = item["coord_values"][item["coord_mask"]].tolist()
        gt_svg = decode_to_svg(reconstruct_svg(skeleton, gt_norm, offset, scale))

        ids = item["input_ids"].unsqueeze(0).to(device)
        attn = item["attention_mask"].unsqueeze(0).to(device)
        seg = item["segment_ids"].unsqueeze(0).to(device)
        cmask = item["coord_mask"].unsqueeze(0).to(device)

        # Inline Euler integration of the learned flow field so we can show a
        # tqdm progress bar per sample.
        B, L = ids.shape
        x = torch.randn(B, L, device=device) * cmask.float()
        dt = 1.0 / num_steps
        for i in tqdm(range(num_steps), desc=f"Sample {row + 1}/{n}", leave=False):
            t = torch.full((B,), i * dt, device=device)
            v = model(ids, attn, seg, x, cmask, t)
            x = x + dt * v * cmask.float()
        pred_norm = x[0][item["coord_mask"]].cpu().tolist()
        pred_svg = decode_to_svg(reconstruct_svg(skeleton, pred_norm, offset, scale))

        label = description[:50] if description else f"Sample {row + 1}"
        axes[row, 0].imshow(_render_svg_to_image(gt_svg))
        axes[row, 0].set_title(f"GT: {label}", fontsize=8)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(_render_svg_to_image(pred_svg))
        axes[row, 1].set_title(f"Pred: {label}", fontsize=8)
        axes[row, 1].axis("off")
        print(f"  sample {row + 1}/{n}: {len(gt_norm)} coords, '{label}'")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of val items to render side-by-side")
    parser.add_argument("--num-val", type=int, default=50,
                        help="Number of val items to load (we sample --num-samples from these)")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Euler integration steps for sampling")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: evaluations/<ckpt_dir>/reconstruction.png)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[eval] Loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model_name = config.get("model_name", "microsoft/Phi-4-mini-instruct")
    lora_r = config.get("lora_r", 32)
    lora_alpha = config.get("lora_alpha", 64)
    lora_targets = config.get("lora_targets",
                              ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
    max_token_len = config.get("max_token_len", 2048)
    print(f"[eval] Step {ckpt.get('global_step', '?')}, "
          f"val_loss={ckpt.get('val_loss', '?')}, model={model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    tokenizer = setup_tokenizer(model_name)
    model = SVGDiffusionModel(
        model_name, tokenizer,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_targets=lora_targets,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if "ema_shadow" in ckpt:
        print("[eval] Applying EMA shadow weights")
        for n, p in model.named_parameters():
            if n in ckpt["ema_shadow"]:
                p.data.copy_(ckpt["ema_shadow"][n].to(p.device, p.dtype))

    print(f"[eval] Loading {args.num_val} validation samples from dataset")
    loader = create_dataloader(
        tokenizer,
        max_samples=args.num_val,
        batch_size=1,
        max_seq_len=1_000_000,
        max_prompt_len=1_000_000,
        max_token_len=max_token_len,
    )
    # Reuse the dataset but with the single-item collate used at train time
    val_loader = torch.utils.data.DataLoader(
        loader.dataset, batch_size=1, shuffle=False, collate_fn=collate_single,
    )

    out_path = args.out
    if out_path is None:
        ckpt_dir = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint)))
        out_path = os.path.join("evaluations", ckpt_dir, "reconstruction.png")

    reconstruct_and_save(
        model, val_loader, device, out_path,
        num_samples=args.num_samples, num_steps=args.num_steps,
    )


if __name__ == "__main__":
    main()
