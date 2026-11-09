"""Session 18: CPU-friendly, two-stage SegFormer-inspired architecture smoke lab.

Educational artificial classes only; does not load research data or official tests.
Real SegFormer has four hierarchical stages and a richer Mix Transformer encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class OverlapPatchEmbed(nn.Module):
    """Overlapping strided convolution followed by token LayerNorm."""
    def __init__(self, in_channels: int, dimension: int, kernel_size: int, stride: int):
        super().__init__()
        if min(in_channels, dimension, kernel_size, stride) < 1 or kernel_size % 2 == 0:
            raise ValueError("Positive channels/stride and an odd kernel size required")
        self.proj = nn.Conv2d(in_channels, dimension, kernel_size, stride, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, x: torch.Tensor):
        features = self.proj(x)
        height, width = features.shape[-2:]
        tokens = self.norm(features.flatten(2).transpose(1, 2))
        return tokens, height, width


class ReducedSelfAttention(nn.Module):
    """All spatial positions are queries; reduced spatial features supply keys/values."""
    def __init__(self, dimension: int, heads: int, reduction: int = 1):
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads or reduction < 1:
            raise ValueError("dimension must divide evenly into heads; reduction >= 1")
        self.dimension = dimension
        self.heads = heads
        self.reduction = reduction
        self.q = nn.Linear(dimension, dimension)
        self.kv = nn.Linear(dimension, 2 * dimension)
        self.proj = nn.Linear(dimension, dimension)
        self.spatial_reduce = (nn.Conv2d(dimension, dimension, reduction, stride=reduction)
                               if reduction > 1 else nn.Identity())
        self.reduce_norm = nn.LayerNorm(dimension) if reduction > 1 else nn.Identity()

    def forward(self, tokens, height: int, width: int, return_attention: bool = False):
        batch, n, dim = tokens.shape
        if n != height * width or dim != self.dimension or min(height, width) < self.reduction:
            raise ValueError("Token grid incompatible with expected feature channels/reduction")
        per_head = dim // self.heads
        queries = self.q(tokens).reshape(batch, n, self.heads, per_head).transpose(1, 2)
        if self.reduction > 1:
            map_ = tokens.transpose(1, 2).reshape(batch, dim, height, width)
            reduced = self.spatial_reduce(map_)
            key_height, key_width = reduced.shape[-2:]
            context = self.reduce_norm(reduced.flatten(2).transpose(1, 2))
        else:
            context = tokens
            key_height, key_width = height, width
        m = context.shape[1]
        key_values = self.kv(context).reshape(batch, m, 2, self.heads, per_head)
        keys, values = key_values.permute(2, 0, 3, 1, 4).unbind(0)
        score = (queries @ keys.transpose(-2, -1)) * (per_head ** -0.5)
        attn = score.softmax(dim=-1)
        out = (attn @ values).transpose(1, 2).reshape(batch, n, dim)
        out = self.proj(out)
        if return_attention:
            return out, attn, (key_height, key_width)
        return out


class MixFeedForward(nn.Module):
    """Pointwise MLP with depthwise 3x3 mixing for local spatial context."""
    def __init__(self, dimension: int, expansion: int = 2):
        super().__init__()
        if dimension < 1 or expansion < 1:
            raise ValueError("Invalid feed-forward width")
        hidden = dimension * expansion
        self.fc1 = nn.Linear(dimension, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dimension)

    def forward(self, tokens, height, width):
        batch, count, _ = tokens.shape
        if count != height * width:
            raise ValueError("MixFFN expects consistent spatial dimensions")
        x = self.fc1(tokens)
        x = x.transpose(1, 2).reshape(batch, -1, height, width)
        x = F.gelu(self.dwconv(x))
        x = x.flatten(2).transpose(1, 2)
        return self.fc2(x)


class MixTransformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, reduction: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.attention = ReducedSelfAttention(dimension, heads, reduction)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = MixFeedForward(dimension)

    def forward(self, tokens, height, width, return_attention=False):
        if return_attention:
            update, attention, key_grid = self.attention(
                self.norm1(tokens), height, width, return_attention=True)
        else:
            update = self.attention(self.norm1(tokens), height, width)
        tokens = tokens + update
        tokens = tokens + self.ffn(self.norm2(tokens), height, width)
        if return_attention:
            return tokens, attention, key_grid
        return tokens


class MiniSegFormer(nn.Module):
    """Two-stage hierarchical transformer + lightweight pointwise decoder.

    Inspired by SegFormer, NOT full MiT/SegFormer-B0/B5, NOT pretrained.
    """
    def __init__(self, classes: int = 2):
        super().__init__()
        if type(classes) is not int or classes < 2:
            raise ValueError("classes must be an integer >= 2")
        self.classes = classes
        self.patch1 = OverlapPatchEmbed(1, 16, kernel_size=7, stride=4)
        self.block1 = MixTransformerBlock(16, heads=2, reduction=2)
        self.patch2 = OverlapPatchEmbed(16, 32, kernel_size=3, stride=2)
        self.block2 = MixTransformerBlock(32, heads=4, reduction=1)
        self.low_projection = nn.Conv2d(16, 16, 1)
        self.high_projection = nn.Conv2d(32, 16, 1)
        self.fusion = nn.Sequential(nn.Conv2d(32, 16, 1), nn.GELU(), nn.Conv2d(16, classes, 1))

    def forward(self, x: torch.Tensor, return_trace: bool = False):
        if (x.ndim != 4 or x.dtype != torch.float32 or x.shape[1] != 1
                or x.shape[0] < 1 or min(x.shape[-2:]) < 16 or not torch.isfinite(x).all()):
            raise ValueError("Expected finite float32 [B,1,H,W], B>=1 and H,W>=16")
        t1, h1, w1 = self.patch1(x)
        if return_trace:
            t1, weights, key_grid = self.block1(t1, h1, w1, return_attention=True)
        else:
            t1 = self.block1(t1, h1, w1)
        f1 = t1.transpose(1, 2).reshape(x.shape[0], 16, h1, w1)
        t2, h2, w2 = self.patch2(f1)
        t2 = self.block2(t2, h2, w2)
        f2 = t2.transpose(1, 2).reshape(x.shape[0], 32, h2, w2)
        low = self.low_projection(f1)
        high = F.interpolate(self.high_projection(f2), size=(h1, w1), mode="bilinear", align_corners=False)
        fusion = torch.cat([low, high], dim=1)
        coarse_logits = self.fusion(fusion)
        logits = F.interpolate(coarse_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if not return_trace:
            return logits
        trace = {"input": list(x.shape), "first_tokens": list(t1.shape),
                 "low_features": list(f1.shape), "second_tokens": list(t2.shape),
                 "high_features": list(f2.shape), "query_tokens": h1 * w1,
                 "key_tokens": key_grid[0] * key_grid[1],
                 "attention": list(weights.shape), "attention_grid": list(key_grid),
                 "decoder_concatenation": list(fusion.shape), "logits": list(logits.shape)}
        # Averaging heads for a selected query gives an illustrative attention distribution.
        center_query = (h1 // 2) * w1 + (w1 // 2)
        center_attention = weights[0, :, center_query, :].mean(dim=0).reshape(key_grid)
        return logits, trace, center_attention


def original_to_binary(mask: np.ndarray):
    """Original 10-class IDs 0..8=pearlite direction, 9=ferrite.
    This conversion is explicit and returns a copy; class 0 is NOT background.
    """
    if (not isinstance(mask, np.ndarray) or mask.ndim != 2 or not mask.size
            or not np.issubdtype(mask.dtype, np.integer)
            or np.any(mask < 0) or np.any(mask > 9)):
        raise ValueError("Expected nonempty original 10-class categorical mask with IDs 0..9")
    return np.where(mask == 9, 0, 1).astype(np.uint8)


def artificial_batch():
    """Exact group-zero geometry/noise recipe from Session 13 for standalone execution."""
    images, targets = [], []
    row, col = np.indices((48, 48))
    for variant in range(2):
        # Match Session 13 __getitem__: regenerate the same source geometry per variant.
        rng = np.random.default_rng(1300)
        center_r, center_c = rng.integers(15, 34, 2)
        radius = int(rng.integers(6, 10))
        binary = ((row - center_r)**2 + (col - center_c)**2 <= radius**2).astype(np.int64)
        clean = np.where(binary, 55.0, 205.0)
        field = np.linspace(.75, 1.15, 48)[None, :] if variant else 1
        gray = np.clip(np.rint(clean * field + rng.normal(0, 5, binary.shape)), 0, 255).astype(np.uint8)
        images.append(torch.from_numpy(gray.copy()).float().unsqueeze(0) / 255)
        targets.append(torch.from_numpy(binary.copy()).long())
    return torch.stack(images), torch.stack(targets)


def smoke_step(seed: int = 18):
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        images, targets = artificial_batch()
        before_images, before_targets = images.clone(), targets.clone()
        model = MiniSegFormer(classes=2).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        criterion = nn.CrossEntropyLoss()
        before_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        logits, trace, attention_map = model(images, return_trace=True)
        initial = criterion(logits, targets)
        first_probability = logits[0].detach().softmax(dim=0)[1].numpy().copy()
        initial.backward()
        grad = {}
        for name, p in model.named_parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError("Invalid or missing gradient: " + name)
            grad[name] = float(p.grad.norm())
        optimizer.step()
        change = {n: float((p.detach() - before_weights[n]).norm()) for n, p in model.named_parameters()}
        if not any(v > 0 for v in change.values()):
            raise RuntimeError("Parameters did not update")
        with torch.no_grad():
            final_logits = model(images)
            after = float(criterion(final_logits, targets))
            ten_shape = list(MiniSegFormer(classes=10)(images[:1]).shape)
            odd_shape = list(model(torch.zeros(1, 1, 65, 67)).shape)
        if not torch.equal(images, before_images) or not torch.equal(targets, before_targets):
            raise RuntimeError("Source tensors were mutated")
        arrays = {"image": images[0, 0].numpy(), "mask": targets[0].numpy(),
                  "probability": first_probability,
                  "prediction": final_logits.argmax(dim=1)[0].numpy(),
                  "attention": attention_map.detach().numpy()}
        report = {"seed": seed, "optimizer": "Adam", "learning_rate": .002, "steps": 1,
                  "trainable_parameters": sum(p.numel() for p in model.parameters()),
                  "initial_loss": float(initial.detach()), "same_batch_loss_after_step": after,
                  "trace": trace, "ten_class_output": ten_shape, "odd_output": odd_shape,
                  "gradient_norms_by_module": {
                      k: sum(v for n, v in grad.items() if n.startswith(k + "."))
                      for k in ("patch1", "block1", "patch2", "block2", "low_projection", "high_projection", "fusion")},
                  "parameter_updates": change, "source_preserved": True,
                  "scope": "Educational two-stage SegFormer-inspired transformer; one artificial optimizer step. Not original SegFormer, not trained real metallography.",
                  "data": "Only deterministic artificial group-zero images; no real train/official test partition accessed.",
                  "original_labels": "Original masks 0..8 pearlite direction, 9 ferrite; derived binary 9->0 and 0..8->1."}
    return report, arrays


def run(output_dir: Path, seed: int = 18):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory exists; select a new path")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report, arrays = smoke_step(seed)
    finally:
        torch.set_num_threads(old_threads)
    report["environment"] = {"python": platform.python_version(),
                             "torch": str(torch.__version__), "numpy": str(np.__version__)}
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), layout="constrained")
    try:
        for ax, name, title in zip(axes, ("image", "mask", "probability", "prediction", "attention"),
                                   ("Artificial grayscale", "Known synthetic mask",
                                    "P(class 1) before update", "Argmax after update",
                                    "Stage-1 center-query attention")):
            ax.imshow(arrays[name], cmap="gray" if name != "attention" else "viridis", interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle("Session 18: MiniSegFormer architecture smoke test (not a trained segmentation benchmark)")
        fig.savefig(output_dir / "session18_transformer_smoke.png", dpi=130)
    finally:
        plt.close(fig)
    (output_dir / "session18_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Parameters:", report["trainable_parameters"])
    print("Loss before:", round(report["initial_loss"], 6), "after one step:", round(report["same_batch_loss_after_step"], 6))
    print("Shapes:", report["trace"]["logits"], report["odd_output"], report["ten_class_output"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/session18_v1"))
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()
    try:
        run(args.output_dir, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
