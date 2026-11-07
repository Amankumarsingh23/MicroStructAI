"""Small same-padded U-Net: artificial CPU architecture and gradient smoke tests."""
import argparse
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from session03_dataset_audit import ROOT, checksum
from session11_pytorch_basics import ToyMicrographDataset, validate_targets


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.refine = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, low, skip):
        up = self.upsample(low)
        trace = {"transposed_shape": list(up.shape), "skip_shape": list(skip.shape),
                 "interpolated": up.shape[-2:] != skip.shape[-2:]}
        if trace["interpolated"]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        combined = torch.cat([skip, up], dim=1)
        trace["aligned_shape"] = list(up.shape)
        trace["concatenated_shape"] = list(combined.shape)
        result = self.refine(combined)
        trace["output_shape"] = list(result.shape)
        return result, trace


class SmallUNet(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
            raise ValueError("Class count must be an integer >=2")
        self.enc1 = DoubleConv(1, 8)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(8, 16)
        self.bottleneck = DoubleConv(16, 32)
        self.up2 = UpBlock(32, 16, 16)
        self.up1 = UpBlock(16, 8, 8)
        self.head = nn.Conv2d(8, classes, 1)

    def forward(self, x, return_trace=False):
        if (x.ndim != 4 or x.shape[1] != 1 or x.dtype != torch.float32
                or x.shape[0] < 1 or min(x.shape[-2:]) < 4):
            raise ValueError("Expected float32 [B,1,H,W], B>=1 and H,W>=4")
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        b = self.bottleneck(p2)
        d2, up2 = self.up2(b, e2)
        d1, up1 = self.up1(d2, e1)
        logits = self.head(d1)
        trace = {"input": list(x.shape), "encoder1": list(e1.shape), "pool1": list(p1.shape),
                 "encoder2": list(e2.shape), "pool2": list(p2.shape), "bottleneck": list(b.shape),
                 "up2": up2, "up1": up1, "logits": list(logits.shape)}
        return (logits, trace) if return_trace else logits


def smoke_step(seed=12):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        images, targets = next(iter(DataLoader(ToyMicrographDataset(), batch_size=2,
                                               shuffle=False, num_workers=0)))
        source_images, source_targets = images.clone(), targets.clone()
        model = SmallUNet().cpu()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        criterion = nn.CrossEntropyLoss()
        before_weights = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer.zero_grad()
        logits, trace = model(images, return_trace=True)
        validate_targets(logits, targets)
        initial_loss = criterion(logits, targets)
        with torch.no_grad():
            probability_before = logits.softmax(1)[0, 1].numpy().copy()
            prediction_before = logits.argmax(1)[0].numpy().copy()
        initial_loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Missing or nonfinite gradient: {name}")
            gradients[name] = float(parameter.grad.norm())
        optimizer.step()
        changes = {name: float((p.detach() - before_weights[name]).norm()) for name, p in model.named_parameters()}
        if not any(v > 0 for v in changes.values()):
            raise RuntimeError("No parameter update occurred")
        with torch.no_grad():
            after = model(images)
            after_loss = float(criterion(after, targets))
            _, odd_trace = model(torch.zeros(1, 1, 65, 67), return_trace=True)
            ten_class = SmallUNet(10)(torch.zeros(1, 1, 64, 64))
        assert torch.equal(images, source_images) and torch.equal(targets, source_targets)
        report = {"seed": seed, "device": "cpu", "steps": 1, "learning_rate": .1,
                  "parameters": sum(p.numel() for p in model.parameters()), "initial_loss": float(initial_loss.detach()),
                  "same_batch_loss_after_step": after_loss, "trace": trace, "odd_size_trace": odd_trace,
                  "ten_class_shape": list(ten_class.shape), "gradient_norms": gradients,
                  "parameter_update_norms": changes,
                  "block_gradient_norm_sums": {block: sum(v for name, v in gradients.items() if name.startswith(block + "."))
                                               for block in ("enc1", "enc2", "bottleneck", "up2", "up1", "head")},
                  "source_tensors_preserved": True,
                  "scope": "One artificial batch and one CPU SGD step, not trained U-Net or a segmentation benchmark",
                  "architecture": "Two pooling levels, same-padded DoubleConv, transposed upsampling, concatenated skips; no normalization/dropout",
                  "fixture": "Reuse Session 11 artificial Dataset; referenced Session 12 package not attached"}
        arrays = {"image": images[0, 0].numpy(), "truth": targets[0].numpy(),
                  "probability_before": probability_before, "probability_after": after.softmax(1)[0, 1].numpy(),
                  "prediction_before": prediction_before, "prediction_after": after.argmax(1)[0].numpy()}
    return report, arrays


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output directory already exists; use a fresh directory")
    report, arrays = smoke_step()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    report["environment"] = {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
                             "cuda_available": torch.cuda.is_available()}
    report["code"] = {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                       "script_sha256": checksum(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    try:
        for ax, (key, title) in zip(axes.ravel(), (
            ("image", "Artificial input"), ("truth", "Known artificial mask"),
            ("probability_before", "Class-1 probability before"), ("probability_after", "Class-1 probability after"),
            ("prediction_before", "Argmax before"), ("prediction_after", "Argmax after"))):
            ax.imshow(arrays[key], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle("Small U-Net architecture smoke test: one step, artificial data, no benchmark")
        fig.savefig(output / "session12_unet_smoke.png", dpi=120)
    finally:
        plt.close(fig)
    (output / "session12_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session12")
    args = parser.parse_args()
    try:
        run(args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
