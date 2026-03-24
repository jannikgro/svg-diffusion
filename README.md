# SVG Diffusion

Text-conditioned SVG coordinate generation via flow matching on a MarkupLM backbone.

The model learns to predict SVG coordinate values using a continuous flow matching objective, conditioned on both a text description and the structural skeleton of an SVG (element types, path commands, attributes). Rather than generating SVG tokens autoregressively, it treats coordinate prediction as a continuous regression problem: given an SVG skeleton with `[COORD]` placeholders, the model fills in the numeric coordinate values by learning a vector field that transports Gaussian noise to the target coordinate distribution.

## Architecture

```
Text prompt ──► Tokenizer ──► Prompt IDs (segment 0)
                                    │
SVG skeleton ──► Tokenizer ──► SVG IDs (segment 1)   ──► MarkupLM + LoRA ──► Hidden states ──► coord_head ──► Predicted velocity
                                    │                          ▲
Noisy coords ──► coord_proj ────────┘                          │
                                                               │
Timestep t ────► SinusoidalEmbedding ──► time_embed ───────────┘
```

**Backbone**: [MarkupLM-base](https://huggingface.co/microsoft/markuplm-base) (~110M params), a BERT-style Transformer pretrained on HTML/XML markup. Chosen because SVG is itself a markup language.

**Adaptation**: LoRA (r=16, alpha=32) applied to `query` and `value` attention projections. The base model is frozen except for LoRA adapters and newly added SVG token embeddings.

**Custom token embeddings**: 56 special tokens are added to the vocabulary:
- 1 `[COORD]` token marking coordinate placeholder positions
- 55 LLM4SVG semantic tokens (`[<|moveto|>]`, `[<|svg_path|>]`, `[<|fill|>]`, etc.)

Only the newly appended embedding rows receive gradients (via a gradient masking hook on the embedding weight).

**Segment embeddings**: A learned 2-way segment embedding distinguishes the text prompt (segment 0) from the SVG skeleton (segment 1).

**Coordinate injection**: Noisy coordinate values are projected to hidden dimension via a 2-layer MLP (`Linear → GELU → Linear`) and added to token embeddings at `[COORD]` positions only (masked by `coord_mask`).

**Time conditioning**: The scalar diffusion timestep `t ∈ [0, 1]` is mapped through a sinusoidal positional embedding followed by a 2-layer MLP, then broadcast-added to all token positions.

**Output head**: A 2-layer MLP (`Linear → GELU → Linear → squeeze`) maps hidden states back to scalar velocity predictions per token position. Only predictions at `[COORD]` positions contribute to the loss.

### Trainable Parameter Breakdown

| Component | Parameters | Notes |
|---|---|---|
| LoRA adapters (query + value, 12 layers) | ~590K | r=16 on 768-dim attention |
| New SVG token embeddings (56 rows) | ~43K | 56 × 768 |
| Segment embedding | ~1.5K | 2 × 768 |
| coord_proj (input MLP) | ~591K | 1 → 768 → 768 |
| time_embed (sinusoidal + MLP) | ~1.2M | 768 → 768 → 768 |
| coord_head (output MLP) | ~591K | 768 → 768 → 1 |
| **Total trainable** | **~3M** | ~2.7% of full model |

## Training Objective: Flow Matching

The model is trained with a **conditional flow matching** (a.k.a. rectified flow) objective:

1. Sample a timestep `t ~ Uniform(0, 1)`
2. Construct noisy coordinates: `x_t = (1 - t) * noise + t * data` (linear interpolation)
3. Target velocity: `v = data - noise` (the straight-line direction from noise to data)
4. Model predicts velocity: `v_hat = model(input_ids, attention_mask, segment_ids, x_t, coord_mask, t)`
5. Loss: MSE between predicted and target velocity, masked to `[COORD]` positions only:
   ```
   L = sum((v_hat - v)^2 * coord_mask) / sum(coord_mask)
   ```

**Sampling** uses Euler integration of the learned velocity field over `num_steps=50` steps, starting from `x_0 ~ N(0, 1)` and integrating to `x_1 ≈ data`.

## Data Pipeline

**Dataset**: [SVGX-SFT-1M](https://huggingface.co/datasets/xingxm/SVGX-SFT-1M) (specifically the `SVGX_SFT_GEN_51k_encode.json` split, ~51K LLM4SVG-encoded SVGs with text descriptions).

**Preprocessing pipeline** (`prepare_dataset.py`):
1. Parse LLM4SVG-encoded SVG string into a **skeleton** (structural tokens with `[COORD]` placeholders) and a **coordinate list** (extracted numeric values)
2. **Per-sample normalization**: coordinates are affine-normalized to `[-1, 1]` using `offset = (max + min) / 2`, `scale = (max - min) / 2`
3. Text prompt is sanitized (reserved SVG tokens escaped) and tokenized (max 64 tokens)
4. SVG skeleton is tokenized; prompt + skeleton are packed into a single sequence (max 512 tokens) with segment IDs
5. `coord_mask` (boolean) and `coord_values` (float) tensors are aligned to token positions
6. If the skeleton is truncated by the 512-token limit, the coordinate list and skeleton are truncated to match

**Collation**: Dynamic padding per batch to the longest sequence, using `pad_token_id` for input_ids and 0/False for masks.

## Configuration

| Parameter | Value |
|---|---|
| Backbone | `microsoft/markuplm-base` |
| Max sequence length | 512 |
| Max prompt length | 64 tokens |
| Max training samples | 25,000 |
| Batch size | 32 |
| Learning rate | 1e-4 (AdamW) |
| Epochs | 1,000 |
| LoRA rank / alpha | 16 / 32 |
| LoRA dropout | 0.1 |
| LoRA targets | `query`, `value` |
| Validation split | 5% |
| Sampling steps (eval) | 50 (Euler) |

## Training Results

Most recent run (25K samples, NVIDIA RTX 6000 Ada, ~5.6 min/epoch):

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.3650 | 0.3066 |
| 5 | 0.2941 | 0.2571 |
| 7 | 0.2300 | 0.1959 |
| 12 | 0.2006 | 0.1880 |

Loss is decreasing steadily. Reconstructions at epoch 12 show vague color blobs and rough shapes but no recognizable SVGs yet. Training was interrupted (KeyboardInterrupt) during epoch 13.

An earlier run loading the full ~514K samples produced anomalous first-epoch metrics (train: 145.8, val: 1259.3), likely from a previous code version before bug fixes.

## Repository Structure

```
├── train_svg_diffusion.py   # Training loop, flow matching loss, evaluation, reconstruction visualization
├── model.py                 # SVGDiffusionModel (MarkupLM + LoRA + coordinate/time heads)
├── svg_utils.py             # LLM4SVG parsing, skeleton extraction, normalization, SVG reconstruction/decoding
├── prepare_dataset.py       # Dataset loading, tokenization, collation; also a standalone visualization script
├── test_svg_diffusion.py    # Pytest suite (parsing, tokenization, model forward/backward, training step)
├── related_work.md          # Literature survey (LLM4SVG, flow matching, diffusion for language)
└── wandb/                   # W&B run logs
```

## Usage

```bash
# Install dependencies
pip install torch transformers peft datasets wandb cairosvg Pillow matplotlib tqdm safetensors

# Run tests
pytest test_svg_diffusion.py -v

# Train
python train_svg_diffusion.py

# Visualize dataset samples (standalone)
python prepare_dataset.py
```

## Bugs, Problems, and Proposed Fixes

### BUG 1: No checkpoint saving — all training progress is lost on interruption

**Severity**: Critical

The training loop saves no model checkpoints. The latest run trained for 12+ epochs (~68 minutes of GPU time) and was lost to a KeyboardInterrupt. With 1,000 target epochs (~93 hours), any crash, OOM, or SLURM timeout will lose everything.

**Fix**: Save checkpoints every N epochs and on best validation loss:
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pt")
if (epoch + 1) % 10 == 0:
    torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")
```

### BUG 2: No learning rate schedule

**Severity**: High

A constant `lr=1e-4` for 1,000 epochs will likely cause the loss to plateau early. The val loss already shows oscillation (0.307 → 0.332 → 0.313 → 0.322) in epochs 1–4 before settling. Without warmup or decay, training is both unstable early and suboptimal late.

**Fix**: Add a cosine schedule with linear warmup:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
```

### BUG 3: No gradient clipping

**Severity**: Medium

Flow matching with Euler integration can produce large velocity targets when `noise` and `data` are far apart. Without gradient clipping, occasional large batches can destabilize training. This may explain the val loss oscillation observed.

**Fix**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### BUG 4: No EMA (Exponential Moving Average)

**Severity**: Medium

Flow matching and diffusion models universally benefit from EMA of model weights for evaluation/sampling. Without EMA, the reconstruction visualizations use the noisy, most-recent training weights, making evaluation quality worse than it could be.

**Fix**: Maintain an EMA copy of the model and use it for `evaluate()` and `reconstruct_samples()`.

### BUG 5: `reconstruct_svg` will crash with `IndexError` if coordinate count mismatches

**Severity**: Medium

In `svg_utils.py:285`, `coordinates[idx]` is accessed without bounds checking. If the model produces fewer coordinates than there are `[COORD]` placeholders in the skeleton (e.g., due to truncation bugs or edge cases), this will crash.

**Fix**: Add a bounds check:
```python
if tok == COORD_TOKEN:
    if idx < len(coordinates):
        result.append(f"{coordinates[idx]:.2f}")
        idx += 1
    else:
        result.append("0.00")  # fallback
```

### BUG 6: Safetensors conversion uses `weights_only=False`

**Severity**: Low (security concern)

In `model.py:21`, `torch.load(bin_path, weights_only=False)` disables the PyTorch pickle safety check (CVE-2025-32434). This is loading from the HuggingFace cache so the risk is limited, but a supply-chain attack on the HuggingFace Hub could exploit this.

**Fix**: Use `weights_only=True` and handle any missing classes explicitly, or use `safetensors` loading directly.

### BUG 7: The full-dataset run produces anomalous loss values

**Severity**: Medium (configuration issue)

The run with ~514K samples showed train loss of 145.8 and val loss of 1259.3 at epoch 1 — orders of magnitude higher than the 25K run. This suggests either (a) the earlier code version had a normalization bug, or (b) some samples in the larger dataset have degenerate coordinate distributions (e.g., a single coordinate, causing `scale=1.0` bypass in normalization while the actual values are very large).

**Fix**: Add dataset-level statistics logging and clamp extreme coordinate values. Filter samples with coordinate ranges that are unreasonably large.

### PROBLEM 8: This is a reconstruction model, not a generative model

**Severity**: Architectural limitation

The model requires the ground-truth SVG skeleton at inference time — it only fills in coordinate values. To generate a novel SVG from text alone, you would also need to generate the skeleton (element types, path commands, attributes, colors). This limits the model to coordinate refinement / inpainting tasks.

**Possible direction**: Two-stage pipeline: (1) autoregressive skeleton generation, (2) flow-matching coordinate prediction. Or: predict both discrete tokens and continuous coordinates jointly.

### PROBLEM 9: Prompt conditioning is very weak

**Severity**: Medium

The text prompt is just segment-0 tokens attending to SVG tokens through the Transformer. With only 64 prompt tokens and no cross-attention or classifier-free guidance, the text signal is diluted. The model may learn to largely ignore the text and just memorize coordinate patterns.

**Possible fix**: Add classifier-free guidance (randomly drop the prompt during training, then scale guidance at sampling time).

### PROBLEM 10: Per-sample normalization leaks information at training time

**Severity**: Low (design concern)

Each sample's coordinates are independently normalized to `[-1, 1]`, and `offset`/`scale` are stored as metadata. At test time, a generated SVG would need these values — but they come from the ground truth. For pure generation, you'd need to either predict offset/scale or use a fixed coordinate space.

### PROBLEM 11: `max_prompt_len=64` is very short

**Severity**: Low

Many descriptions in the SVGX dataset are detailed (e.g., "SVG illustration of a construction worker holding a hammer..."). At 64 subword tokens, these get heavily truncated, reducing the model's ability to condition on text.

**Fix**: Increase `max_prompt_len` to 128 and reduce `max_seq_len` budget for the SVG portion, or increase `max_seq_len` to 768.

## Verdict: Will This Training Work?

**The training loop runs and the loss decreases — the model is learning something.** The flow matching formulation is mathematically correct, the data pipeline roundtrips cleanly (verified by the 100-sample roundtrip test), and the architecture is sound in principle.

**However, the current setup is unlikely to produce high-quality SVG reconstructions.** After 12 epochs of training, the reconstructions show only vague color blobs with no recognizable structure. The specific reasons:

1. **The loss plateau is still high.** Train loss of 0.20 means the average coordinate velocity error is substantial. For SVGs with dozens to hundreds of coordinates, small per-coordinate errors compound into unrecognizable shapes. The model likely needs loss in the 0.01–0.05 range to produce coherent output.

2. **The model capacity is limited.** Only ~3M trainable parameters (LoRA + heads) on a 110M frozen backbone. The coordinate prediction task requires learning precise spatial relationships that the frozen MarkupLM weights were never trained for. Unfreezing more layers or increasing LoRA rank would help.

3. **No checkpoint saving means the 1,000-epoch target is unreachable.** At ~5.6 min/epoch, the full training would take ~93 hours. Without checkpointing, any interruption loses everything. On a SLURM cluster with job time limits, this is almost certain to fail.

4. **Missing standard training infrastructure** (LR schedule, gradient clipping, EMA, checkpointing) will limit final quality even if training completes.

**Bottom line: The approach is scientifically reasonable and the implementation is mostly correct, but the training will not converge to usable quality without adding checkpointing, a learning rate schedule, gradient clipping, and likely more model capacity. The loss needs to drop by roughly another order of magnitude, which will require both more epochs and the training stability improvements listed above. Fix the critical bugs first (especially checkpointing), then run a longer experiment.**
