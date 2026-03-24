import math
import torch
import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from svg_utils import COORD_TOKEN


class SinusoidalEmbedding(nn.Module):
    """Maps a scalar timestep to a sinusoidal positional embedding vector."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.unsqueeze(-1).float() * freqs
        return torch.cat([args.cos(), args.sin()], dim=-1)


class SVGDiffusionModel(nn.Module):
    def __init__(self, model_name_or_config, tokenizer, lora_r=32, lora_alpha=64,
                 lora_dropout=0.1, lora_targets=None):
        super().__init__()
        if isinstance(model_name_or_config, str):
            base = AutoModel.from_pretrained(
                model_name_or_config, dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            base.gradient_checkpointing_enable()
            if lora_targets is None:
                lora_targets = ["qkv_proj"]
        else:
            base = AutoModel.from_config(model_name_or_config)
            if lora_targets is None:
                lora_targets = ["query", "value"]

        original_vocab_size = base.get_input_embeddings().num_embeddings
        base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        h = base.config.hidden_size

        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_targets, lora_dropout=lora_dropout, bias="none")
        self.encoder = get_peft_model(base, lora_config)
        self.coord_token_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
        self.svg_token_start = original_vocab_size

        # PEFT freezes the base embedding table, but our SVG tokens were just
        # appended and would otherwise stay random forever. Re-enable gradients
        # and mask them so only the newly added rows are updated.
        embedding = self.encoder.get_input_embeddings()
        embedding.weight.requires_grad = True
        trainable_rows = torch.zeros(embedding.num_embeddings, 1)
        trainable_rows[self.svg_token_start:] = 1.0
        self.register_buffer("trainable_embedding_rows", trainable_rows)
        embedding.weight.register_hook(lambda grad: grad * self.trainable_embedding_rows.to(grad.dtype))

        self.segment_embed = nn.Embedding(2, h)
        self.coord_proj = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.time_embed = nn.Sequential(SinusoidalEmbedding(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))

        # Control tokens: learnable base embedding + sinusoidal position encoding
        # of the source coordinate position in the input sequence.
        self.ctrl_token_embed = nn.Parameter(torch.randn(h) * 0.02)
        self.ctrl_pos_embed = SinusoidalEmbedding(h)

        self.coord_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def _build_control_tokens(self, coord_mask, timestep):
        """Build control token embeddings, padded to the max coord count in the batch.

        Returns (ctrl_embeds, ctrl_attn, max_n):
            ctrl_embeds: (B, max_n, h)
            ctrl_attn:   (B, max_n) attention mask (1 for real, 0 for padding)
            max_n:       int, max number of coord tokens across the batch
        """
        B = coord_mask.shape[0]
        device = coord_mask.device
        h = self.ctrl_token_embed.shape[0]
        n_coords = coord_mask.sum(dim=1)  # (B,)
        max_n = n_coords.max().item()

        ctrl_list = []
        attn_list = []
        for b in range(B):
            positions = coord_mask[b].nonzero(as_tuple=True)[0].float()  # (n_b,)
            n = positions.shape[0]
            ctrl = self.ctrl_token_embed.unsqueeze(0).expand(n, -1)    # (n, h)
            ctrl = ctrl + self.ctrl_pos_embed(positions)                # + source position
            ctrl = ctrl + self.time_embed(timestep[b:b+1]).expand(n, -1)  # + timestep
            # Pad to max_n
            if n < max_n:
                ctrl = torch.cat([ctrl, torch.zeros(max_n - n, h, device=device)], dim=0)
            ctrl_list.append(ctrl)
            attn = torch.zeros(max_n, device=device)
            attn[:n] = 1.0
            attn_list.append(attn)

        ctrl_embeds = torch.stack(ctrl_list)        # (B, max_n, h)
        ctrl_attn = torch.stack(attn_list).long()   # (B, max_n)
        return ctrl_embeds, ctrl_attn, max_n

    def forward(self, input_ids, attention_mask, segment_ids, noisy_coords, coord_mask, timestep):
        B, L = input_ids.shape

        # Input embeddings with coordinate and time injection (unchanged)
        embeds = self.encoder.get_input_embeddings()(input_ids)
        embeds = embeds + self.segment_embed(segment_ids)
        embeds = embeds + self.coord_proj(noisy_coords.unsqueeze(-1)) * coord_mask.unsqueeze(-1).float()
        embeds = embeds + self.time_embed(timestep).unsqueeze(1)

        # Build and append control tokens
        ctrl_embeds, ctrl_attn, max_n = self._build_control_tokens(coord_mask, timestep)
        if max_n == 0:
            return torch.zeros(B, L, device=input_ids.device)

        full_embeds = torch.cat([embeds, ctrl_embeds], dim=1)          # (B, L+max_n, h)
        full_attn = torch.cat([attention_mask, ctrl_attn], dim=1)      # (B, L+max_n)

        # Cast to backbone dtype (bfloat16 for Phi-4-mini) — custom heads are
        # float32 but the frozen backbone expects its native precision.
        backbone_dtype = self.encoder.get_input_embeddings().weight.dtype
        hidden = self.encoder(inputs_embeds=full_embeds.to(backbone_dtype), attention_mask=full_attn).last_hidden_state
        ctrl_hidden = hidden[:, L:].float()                             # (B, max_n, h) back to float32
        ctrl_pred = self.coord_head(ctrl_hidden).squeeze(-1)            # (B, max_n)

        # Scatter control-token predictions back to (B, L) — differentiable
        indices = coord_mask.long().cumsum(dim=1) - 1   # maps each position to its coord index
        indices = indices.clamp(min=0)                   # non-coord positions clamp to 0
        out = ctrl_pred.gather(1, indices) * coord_mask.float()  # zero out non-coord positions
        return out

    @torch.no_grad()
    def sample(self, input_ids, attention_mask, segment_ids, coord_mask, num_steps=50):
        """Generate coordinates via Euler integration of the learned flow field."""
        B, L = input_ids.shape
        x = torch.randn(B, L, device=input_ids.device) * coord_mask.float()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=input_ids.device)
            v = self(input_ids, attention_mask, segment_ids, x, coord_mask, t)
            x = x + dt * v * coord_mask.float()
        return x
