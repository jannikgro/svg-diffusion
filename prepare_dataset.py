import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from svg_utils import COORD_TOKEN, SVG_SEMANTIC_TOKENS, parse_encoded_svg, normalize_coordinates


def setup_tokenizer(model_name="microsoft/markuplm-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = [COORD_TOKEN] + SVG_SEMANTIC_TOKENS
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return tokenizer


def _sanitize_prompt_text(prompt_text):
    """Prevent reserved SVG tokens from appearing in the prompt segment."""
    sanitized = prompt_text or ""
    for token in [COORD_TOKEN] + SVG_SEMANTIC_TOKENS:
        if token in sanitized:
            sanitized = sanitized.replace(token, token.replace("[", "(").replace("]", ")"))
    return sanitized


def _build_prompt_svg_pair(tokenizer, prompt_ids, svg_ids, max_seq_len):
    """Build a prompt-first / SVG-second pair sequence and explicit segment IDs."""
    special_tokens = tokenizer.num_special_tokens_to_add(pair=True)
    prompt_budget = max(0, max_seq_len - special_tokens)
    prompt_ids = prompt_ids[:prompt_budget]
    svg_budget = max_seq_len - len(prompt_ids) - special_tokens
    svg_ids = svg_ids[:max(0, svg_budget)]

    input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, svg_ids)
    attention_mask = [1] * len(input_ids)
    special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)

    segment_ids = []
    prompt_tokens_seen = 0
    svg_started = False
    for is_special in special_mask:
        if is_special:
            segment_ids.append(1 if svg_started else 0)
        elif prompt_tokens_seen < len(prompt_ids):
            segment_ids.append(0)
            prompt_tokens_seen += 1
        else:
            svg_started = True
            segment_ids.append(1)

    return input_ids, attention_mask, segment_ids


def process_svg_sample(svg_text, tokenizer, max_seq_len=512, description="", max_prompt_len=64):
    """Process a single encoded SVG string into tensors for training.

    Encodes the prompt as the first segment and the SVG skeleton as the second.
    Returns a dict with input_ids, attention_mask, segment_ids, coord_mask, and
    coord_values, or None if the sample should be skipped.
    """
    skeleton, coords = parse_encoded_svg(svg_text)
    if not coords:
        return None

    norm_coords, offset, scale = normalize_coordinates(coords)
    prompt_text = _sanitize_prompt_text(description)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, truncation=True, max_length=max_prompt_len)
    svg_ids = tokenizer.encode(skeleton, add_special_tokens=False)
    input_ids, attention_mask, segment_ids = _build_prompt_svg_pair(tokenizer, prompt_ids, svg_ids, max_seq_len)

    coord_token_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
    num_coords_in_ids = sum(1 for tid in input_ids if tid == coord_token_id)
    if num_coords_in_ids == 0:
        return None
    norm_coords = norm_coords[:num_coords_in_ids]

    coord_mask, coord_values, ci = [], [], 0
    for tid in input_ids:
        if tid == coord_token_id and ci < len(norm_coords):
            coord_mask.append(1)
            coord_values.append(norm_coords[ci])
            ci += 1
        else:
            coord_mask.append(0)
            coord_values.append(0.0)

    # Truncate skeleton to match the number of [COORD] tokens that survived
    # tokenizer truncation, so reconstruct_svg won't run out of coordinates.
    if num_coords_in_ids < skeleton.split().count(COORD_TOKEN):
        skel_tokens = skeleton.split()
        coord_seen = 0
        for trunc_idx, st in enumerate(skel_tokens):
            if st == COORD_TOKEN:
                coord_seen += 1
                if coord_seen == num_coords_in_ids:
                    skel_tokens = skel_tokens[:trunc_idx + 1]
                    break
        skeleton = " ".join(skel_tokens)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "coord_mask": coord_mask,
        "coord_values": coord_values,
        "coord_offset": offset,
        "coord_scale": scale,
        "skeleton": skeleton,
        "description": description,
    }


class SVGDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids": torch.tensor(s["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(s["attention_mask"], dtype=torch.long),
            "segment_ids": torch.tensor(s["segment_ids"], dtype=torch.long),
            "coord_mask": torch.tensor(s["coord_mask"], dtype=torch.bool),
            "coord_values": torch.tensor(s["coord_values"], dtype=torch.float32),
            "coord_offset": torch.tensor(s["coord_offset"], dtype=torch.float32),
            "coord_scale": torch.tensor(s["coord_scale"], dtype=torch.float32),
            "skeleton": s["skeleton"],
            "description": s["description"],
        }


def make_collate_fn(pad_token_id):
    """Create a collate function that pads batches to equal length.

    Uses pad_token_id for input_ids, 0 for attention_mask, segment_ids, and
    coord_mask, and 0.0 for coord_values. coord_offset and coord_scale are
    per-sample scalars and are simply stacked.
    """
    def collate_fn(batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        out = {"input_ids": [], "attention_mask": [], "segment_ids": [], "coord_mask": [], "coord_values": []}
        for b in batch:
            pad = max_len - b["input_ids"].shape[0]
            out["input_ids"].append(F.pad(b["input_ids"], (0, pad), value=pad_token_id))
            out["attention_mask"].append(F.pad(b["attention_mask"], (0, pad), value=0))
            out["segment_ids"].append(F.pad(b["segment_ids"], (0, pad), value=0))
            out["coord_mask"].append(F.pad(b["coord_mask"], (0, pad), value=False))
            out["coord_values"].append(F.pad(b["coord_values"], (0, pad), value=0.0))
        out = {k: torch.stack(v) for k, v in out.items()}
        out["coord_offset"] = torch.stack([b["coord_offset"] for b in batch])
        out["coord_scale"] = torch.stack([b["coord_scale"] for b in batch])
        out["skeleton"] = [b["skeleton"] for b in batch]
        out["description"] = [b["description"] for b in batch]
        return out
    return collate_fn


def create_dataloader(tokenizer, dataset_name="xingxm/SVGX-SFT-1M", data_file="SVGX_SFT_GEN_51k_encode.json", split="train", max_samples=None, max_seq_len=512, max_prompt_len=64, batch_size=8, num_workers=0):
    from tqdm import tqdm
    hf_dataset = load_dataset(dataset_name, split=split, data_files=data_file, streaming=True)
    samples = []
    for i, item in enumerate(tqdm(hf_dataset, total=max_samples, desc="Loading SVGs")):
        if max_samples and i >= max_samples:
            break
        svg_text = item.get("output", "")
        description = item.get("input", "")
        processed = process_svg_sample(svg_text, tokenizer, max_seq_len=max_seq_len, description=description, max_prompt_len=max_prompt_len)
        if processed is not None:
            samples.append(processed)
    print(f"Loaded {len(samples)} valid samples (from {i + 1} total)")

    dataset = SVGDataset(samples)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)


if __name__ == "__main__":
    import io
    import matplotlib.pyplot as plt
    from PIL import Image
    import cairosvg
    from svg_utils import (
        parse_encoded_svg, normalize_coordinates, reconstruct_svg, decode_to_svg,
    )

    NUM_EXAMPLES = 6

    def render_svg(svg_xml, size=256):
        """Render SVG XML to a PIL Image, grey fallback on failure."""
        try:
            png = cairosvg.svg2png(bytestring=svg_xml.encode("utf-8"),
                                   output_width=size, output_height=size)
            return Image.open(io.BytesIO(png)).convert("RGB")
        except Exception:
            return Image.new("RGB", (size, size), (200, 200, 200))

    print(f"Loading {NUM_EXAMPLES} examples from SVGX-SFT-1M...")
    raw_dataset = load_dataset(
        "xingxm/SVGX-SFT-1M", split="train",
        data_files="SVGX_SFT_GEN_51k.json", streaming=True,
    )
    enc_dataset = load_dataset(
        "xingxm/SVGX-SFT-1M", split="train",
        data_files="SVGX_SFT_GEN_51k_encode.json", streaming=True,
    )

    examples = []
    seen_svgs = set()
    for raw_item, enc_item in zip(raw_dataset, enc_dataset):
        svg_xml = raw_item.get("output", "")
        if svg_xml.strip().startswith("<svg") and svg_xml not in seen_svgs:
            seen_svgs.add(svg_xml)
            examples.append({
                "svg": svg_xml,
                "encoded": enc_item.get("output", ""),
                "description": raw_item.get("input", ""),
            })
        if len(examples) >= NUM_EXAMPLES:
            break

    print(f"Loaded {len(examples)} examples.\n")

    fig, axes = plt.subplots(len(examples), 2, figsize=(6, 3 * len(examples)))

    for i, ex in enumerate(examples):
        svg_xml = ex["svg"]
        encoded = ex["encoded"]
        description = ex["description"]

        # Right: encode → parse → normalize → denormalize → reconstruct → decode
        skeleton, coords = parse_encoded_svg(encoded)
        norm_coords, offset, scale = normalize_coordinates(coords)
        reconstructed_encoded = reconstruct_svg(skeleton, norm_coords, offset, scale)
        recon_xml = decode_to_svg(reconstructed_encoded)

        print(f"--- Example {i + 1} ---")
        print(f"  Description: {description}")
        print(f"  SVG length: {len(svg_xml)} chars")
        print(f"  SVG:\n{svg_xml}")
        print()

        gt_img = render_svg(svg_xml)
        recon_img = render_svg(recon_xml)

        axes[i, 0].imshow(gt_img)
        axes[i, 0].set_title(f"Original: {description[:40]}", fontsize=8)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon_img)
        axes[i, 1].set_title("Reconstructed", fontsize=8)
        axes[i, 1].axis("off")

    plt.suptitle("Original vs Reconstructed SVGs", fontsize=12)
    plt.tight_layout()
    plt.savefig("example_svgs.png", dpi=150)
    print("Saved plot to example_svgs.png")
    plt.show()
