import math
import torch
import torch.nn as nn
from transformers import MarkupLMModel, MarkupLMConfig
from peft import LoraConfig, get_peft_model
from svg_utils import COORD_TOKEN


def _load_markuplm(model_name):
    """Load MarkupLMModel, converting to safetensors on the fly if needed
    (works around CVE-2025-32434 torch.load restriction in transformers >=5
    when only pytorch_model.bin is available on the hub).
    """
    import logging, warnings, os
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    bin_path = hf_hub_download(model_name, "pytorch_model.bin")
    sf_path = os.path.join(os.path.dirname(bin_path), "model.safetensors")
    if not os.path.exists(sf_path):
        sd = torch.load(bin_path, map_location="cpu", weights_only=False)
        sd = {k: v.contiguous() for k, v in sd.items() if k.startswith("markuplm.")}
        save_file(sd, sf_path)

    import transformers
    prev_level = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    logging.disable(logging.WARNING)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = MarkupLMModel.from_pretrained(os.path.dirname(bin_path), use_safetensors=True, torch_dtype=torch.float32)
    transformers.logging.set_verbosity(prev_level)
    logging.disable(logging.NOTSET)
    return model


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
    def __init__(self, model_name_or_config, tokenizer, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        if isinstance(model_name_or_config, MarkupLMConfig):
            base = MarkupLMModel(model_name_or_config)
        else:
            base = _load_markuplm(model_name_or_config)
        base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        h = base.config.hidden_size

        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["query", "value"], lora_dropout=lora_dropout, bias="none")
        self.encoder = get_peft_model(base, lora_config)
        self.coord_token_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)

        self.coord_proj = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))
        self.time_embed = nn.Sequential(SinusoidalEmbedding(h), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.coord_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask, noisy_coords, coord_mask, timestep):
        embeds = self.encoder.get_input_embeddings()(input_ids)
        embeds = embeds + self.coord_proj(noisy_coords.unsqueeze(-1)) * coord_mask.unsqueeze(-1).float()
        embeds = embeds + self.time_embed(timestep).unsqueeze(1)
        hidden = self.encoder(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        return self.coord_head(hidden).squeeze(-1)

    @torch.no_grad()
    def sample(self, input_ids, attention_mask, coord_mask, num_steps=50):
        """Generate coordinates via Euler integration of the learned flow field."""
        B, L = input_ids.shape
        x = torch.randn(B, L, device=input_ids.device) * coord_mask.float()
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=input_ids.device)
            v = self(input_ids, attention_mask, x, coord_mask, t)
            x = x + dt * v * coord_mask.float()
        return x
