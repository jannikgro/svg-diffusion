import pytest
import torch
from transformers import MarkupLMConfig
from svg_utils import COORD_TOKEN, parse_encoded_svg, normalize_coordinates, denormalize_coordinates, reconstruct_svg, _tokenize_encoded_svg
from prepare_dataset import setup_tokenizer, process_svg_sample, SVGDataset, make_collate_fn
from model import SVGDiffusionModel, SinusoidalEmbedding
from train_svg_diffusion import flow_matching_loss

SAMPLE_SVG = "[<|START_OF_SVG|>][<|svg_path|>][<|fill|>]#FF0000[<|path_d|>][<|moveto|>]10.5 20.3[<|lineto|>]30 40[<|close_the_path|>][<|END_OF_SVG|>]"
SHORT_SVG = "[<|START_OF_SVG|>][<|svg_circle|>][<|cx|>]50[<|cy|>]50[<|r|>]40[<|fill|>]red[<|END_OF_SVG|>]"
COMPLEX_SVG = "[<|START_OF_SVG|>][<|svg_path|>][<|path_d|>][<|moveto|>]79.3 120[<|curveto|>]0 2.21 -6.85 4 -15.3 4[<|smooth_curveto|>]-15.3 -1.79 -15.3 -4 6.85 -4.01 15.3 -4.01 15.3 1.79 15.3 4.01[<|close_the_path|>][<|fill|>]#504f4f[<|END_OF_SVG|>]"


class TestTokenizer:
    def test_tokenize_concatenated(self):
        tokens = _tokenize_encoded_svg("[<|moveto|>]10.5 20.3[<|lineto|>]30")
        assert tokens == ["[<|moveto|>]", "10.5", "20.3", "[<|lineto|>]", "30"]

    def test_tokenize_color(self):
        tokens = _tokenize_encoded_svg("[<|fill|>]#FF0000[<|END_OF_SVG|>]")
        assert tokens == ["[<|fill|>]", "#FF0000", "[<|END_OF_SVG|>]"]

    def test_tokenize_negative(self):
        tokens = _tokenize_encoded_svg("[<|curveto|>]-6.85 4 -15.3 4")
        assert tokens == ["[<|curveto|>]", "-6.85", "4", "-15.3", "4"]


class TestSVGParsing:
    def test_parse_coords(self):
        skeleton, coords = parse_encoded_svg(SAMPLE_SVG)
        assert coords == [10.5, 20.3, 30.0, 40.0]
        assert skeleton.count(COORD_TOKEN) == 4

    def test_colors_preserved(self):
        skeleton, _ = parse_encoded_svg(SAMPLE_SVG)
        assert "#FF0000" in skeleton

    def test_circle_attrs(self):
        skeleton, coords = parse_encoded_svg(SHORT_SVG)
        assert coords == [50.0, 50.0, 40.0]
        assert "red" in skeleton

    def test_complex_path(self):
        skeleton, coords = parse_encoded_svg(COMPLEX_SVG)
        assert coords[0] == 79.3
        assert coords[1] == 120.0
        assert -15.3 in coords
        assert "#504f4f" in skeleton

    def test_non_coord_after_fill(self):
        svg = "[<|START_OF_SVG|>][<|svg_rect|>][<|fill-opacity|>]0.5[<|x|>]10[<|END_OF_SVG|>]"
        skeleton, coords = parse_encoded_svg(svg)
        assert coords == [10.0]
        assert "0.5" in skeleton

    def test_normalize_denormalize_roundtrip(self):
        coords = [10.0, -20.0, 150.0]
        normed, offset, scale = normalize_coordinates(coords)
        assert all(-1.0 <= c <= 1.0 for c in normed)
        denormed = denormalize_coordinates(normed, offset, scale)
        for a, b in zip(coords, denormed):
            assert abs(a - b) < 1e-6

    def test_reconstruct(self):
        skeleton, coords = parse_encoded_svg(SHORT_SVG)
        reconstructed = reconstruct_svg(skeleton, coords)
        assert "[COORD]" not in reconstructed
        assert "50.00" in reconstructed
        assert "[<|cx|>]" in reconstructed

    def test_empty_svg(self):
        skeleton, coords = parse_encoded_svg("[<|START_OF_SVG|>][<|END_OF_SVG|>]")
        assert coords == []

    def test_multiple_path_commands(self):
        svg = "[<|START_OF_SVG|>][<|svg_path|>][<|path_d|>][<|moveto|>]1 2[<|curveto|>]3 4 5 6 7 8[<|close_the_path|>][<|END_OF_SVG|>]"
        skeleton, coords = parse_encoded_svg(svg)
        assert coords == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        assert skeleton.count(COORD_TOKEN) == 8


@pytest.fixture(scope="module")
def tokenizer():
    return setup_tokenizer()


class TestHFTokenizer:
    def test_coord_token_registered(self, tokenizer):
        assert COORD_TOKEN in tokenizer.get_added_vocab()

    def test_semantic_tokens_registered(self, tokenizer):
        vocab = tokenizer.get_added_vocab()
        assert "[<|moveto|>]" in vocab
        assert "[<|END_OF_SVG|>]" in vocab

    def test_coord_token_preserved_in_encoding(self, tokenizer):
        coord_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
        text = f"hello {COORD_TOKEN} world {COORD_TOKEN}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert ids.count(coord_id) == 2

    def test_semantic_token_single_id(self, tokenizer):
        tok = "[<|moveto|>]"
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        ids = tokenizer.encode(tok, add_special_tokens=False)
        assert ids == [tok_id]


class TestProcessSample:
    def test_basic(self, tokenizer):
        result = process_svg_sample(SAMPLE_SVG, tokenizer)
        assert result is not None
        assert sum(result["coord_mask"]) == 4

    def test_circle(self, tokenizer):
        result = process_svg_sample(SHORT_SVG, tokenizer)
        assert result is not None
        assert sum(result["coord_mask"]) == 3

    def test_coord_values_at_mask_positions(self, tokenizer):
        result = process_svg_sample(SAMPLE_SVG, tokenizer)
        for mask_val, coord_val in zip(result["coord_mask"], result["coord_values"]):
            if not mask_val:
                assert coord_val == 0.0

    def test_skip_empty_coords(self, tokenizer):
        result = process_svg_sample("[<|START_OF_SVG|>][<|END_OF_SVG|>]", tokenizer)
        assert result is None


class TestDataset:
    def test_dataset_len(self, tokenizer):
        s = process_svg_sample(SAMPLE_SVG, tokenizer)
        ds = SVGDataset([s])
        assert len(ds) == 1

    def test_dataset_tensors(self, tokenizer):
        s = process_svg_sample(SAMPLE_SVG, tokenizer)
        ds = SVGDataset([s])
        item = ds[0]
        assert item["input_ids"].dtype == torch.long
        assert item["coord_mask"].dtype == torch.bool
        assert item["coord_values"].dtype == torch.float32

    def test_collate_padding(self, tokenizer):
        s1 = process_svg_sample(SAMPLE_SVG, tokenizer)
        s2 = process_svg_sample(SHORT_SVG, tokenizer)
        ds = SVGDataset([s1, s2])
        collate_fn = make_collate_fn(tokenizer.pad_token_id)
        batch = collate_fn([ds[0], ds[1]])
        assert batch["input_ids"].shape[0] == 2
        expected_len = max(len(s1["input_ids"]), len(s2["input_ids"]))
        assert batch["input_ids"].shape[1] == expected_len
        assert batch["attention_mask"][1, -1].item() == 0


@pytest.fixture(scope="module")
def small_config():
    return MarkupLMConfig(vocab_size=200, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, max_position_embeddings=128)


def _make_batch(tokenizer, B=2, L=20):
    coord_id = tokenizer.convert_tokens_to_ids(COORD_TOKEN)
    input_ids = torch.randint(0, 100, (B, L))
    input_ids[:, 5] = coord_id
    input_ids[:, 10] = coord_id
    attention_mask = torch.ones(B, L, dtype=torch.long)
    coord_mask = (input_ids == coord_id)
    coord_values = torch.rand(B, L) * coord_mask.float()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "coord_mask": coord_mask, "coord_values": coord_values}


class TestModel:
    def test_sinusoidal(self):
        emb = SinusoidalEmbedding(64)
        out = emb(torch.tensor([0.0, 0.5, 1.0]))
        assert out.shape == (3, 64)

    def test_forward_shape(self, small_config, tokenizer):
        model = SVGDiffusionModel(small_config, tokenizer)
        batch = _make_batch(tokenizer)
        pred = model(batch["input_ids"], batch["attention_mask"], batch["coord_values"], batch["coord_mask"], torch.rand(2))
        assert pred.shape == (2, 20)

    def test_sample_shape(self, small_config, tokenizer):
        model = SVGDiffusionModel(small_config, tokenizer)
        batch = _make_batch(tokenizer, B=1, L=15)
        result = model.sample(batch["input_ids"], batch["attention_mask"], batch["coord_mask"], num_steps=3)
        assert result.shape == (1, 15)

    def test_lora_params_trainable(self, small_config, tokenizer):
        model = SVGDiffusionModel(small_config, tokenizer)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert any("lora" in n for n in trainable)
        assert any("coord_proj" in n for n in trainable)
        assert any("coord_head" in n for n in trainable)
        assert any("time_embed" in n for n in trainable)

    def test_base_frozen(self, small_config, tokenizer):
        model = SVGDiffusionModel(small_config, tokenizer)
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        assert any("word_embeddings" in n for n in frozen)


class TestTraining:
    def test_flow_matching_loss(self, small_config, tokenizer):
        model = SVGDiffusionModel(small_config, tokenizer)
        batch = _make_batch(tokenizer)
        loss = flow_matching_loss(model, batch, "cpu")
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_training_step_reduces_loss(self, small_config, tokenizer):
        torch.manual_seed(0)
        model = SVGDiffusionModel(small_config, tokenizer)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        batch = _make_batch(tokenizer)
        losses = []
        for _ in range(20):
            loss = flow_matching_loss(model, batch, "cpu")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5

    def test_end_to_end_with_real_svg(self, small_config, tokenizer):
        s1 = process_svg_sample(SAMPLE_SVG, tokenizer)
        s2 = process_svg_sample(SHORT_SVG, tokenizer)
        ds = SVGDataset([s1, s2])
        collate_fn = make_collate_fn(tokenizer.pad_token_id)
        batch = collate_fn([ds[0], ds[1]])

        model = SVGDiffusionModel(small_config, tokenizer)
        loss = flow_matching_loss(model, batch, "cpu")
        assert not torch.isnan(loss)

        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        loss.backward()
        optimizer.step()


class TestReconstructRoundtrip:
    """Load 100 real samples from the dataset, encode and decode them,
    and verify the reconstructed encoded SVG matches the original."""

    def test_roundtrip_100_samples(self):
        from datasets import load_dataset

        ds = load_dataset(
            "xingxm/SVGX-SFT-1M",
            split="train",
            data_files="SVGX_SFT_GEN_51k_encode.json",
            streaming=True,
        )

        checked = 0
        for item in ds:
            original = item.get("output", "")
            skeleton, coords = parse_encoded_svg(original)
            if not coords:
                continue

            # Normalize then reconstruct
            norm_coords, offset, scale = normalize_coordinates(coords)
            reconstructed = reconstruct_svg(skeleton, norm_coords, offset, scale)

            # Re-parse the reconstruction and compare structurally
            recon_skeleton, recon_coords = parse_encoded_svg(reconstructed)
            assert skeleton == recon_skeleton, (
                f"Skeleton mismatch on sample {checked}"
            )
            assert len(coords) == len(recon_coords), (
                f"Coord count mismatch on sample {checked}: "
                f"{len(coords)} vs {len(recon_coords)}"
            )
            for j, (a, b) in enumerate(zip(coords, recon_coords)):
                assert abs(a - b) < 0.01, (
                    f"Coord {j} mismatch on sample {checked}: {a} vs {b}"
                )

            checked += 1
            if checked >= 100:
                break

        assert checked == 100, f"Only found {checked} valid samples (need 100)"
