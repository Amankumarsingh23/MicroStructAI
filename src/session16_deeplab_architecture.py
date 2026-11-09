"""CPU-friendly DeepLab-style architecture with traced ASPP and one-step checks."""
import argparse
import json
from pathlib import Path
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
from session11_pytorch_basics import validate_targets
from session13_training_lab import GroupedToyDataset


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, stride=1):
        super().__init__()
        if not isinstance(dilation, int) or dilation < 1 or not isinstance(stride, int) or stride < 1:
            raise ValueError("Dilation and stride must be positive integers")
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                                   padding=dilation, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.pointwise(self.depthwise(x)))


class MiniASPP(nn.Module):
    def __init__(self, in_channels=32, branch_channels=8, out_channels=16):
        super().__init__()
        self.point_branch = nn.Sequential(nn.Conv2d(in_channels, branch_channels, 1), nn.ReLU())
        self.dilated_branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, branch_channels, 3, dilation=rate, padding=rate), nn.ReLU())
            for rate in (1, 2, 3)
        ])
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(in_channels, branch_channels, 1), nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(branch_channels * 5, out_channels, 1), nn.ReLU())

    def forward(self, x, return_trace=False):
        branches = [self.point_branch(x)]
        branches.extend(branch(x) for branch in self.dilated_branches)
        pooled = self.global_branch(x)
        branches.append(F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False))
        combined = torch.cat(branches, dim=1)
        result = self.project(combined)
        trace = {"branch_shapes": [list(branch.shape) for branch in branches],
                 "global_pooled_shape": list(pooled.shape), "concatenated_shape": list(combined.shape),
                 "projection_shape": list(result.shape)}
        return (result, trace) if return_trace else result


class MiniDeepLabV3Plus(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
            raise ValueError("Class count must be an integer of at least two")
        self.low_encoder = nn.Sequential(nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU())
        self.high_encoder = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
                                          DepthwiseSeparableConv(16, 32))
        self.aspp = MiniASPP()
        self.low_projection = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU())
        self.decoder = DepthwiseSeparableConv(20, 16)
        self.head = nn.Conv2d(16, classes, 1)

    def forward(self, x, return_trace=False):
        if (x.ndim != 4 or x.dtype != torch.float32 or x.shape[1] != 1
                or x.shape[0] < 1 or min(x.shape[-2:]) < 4):
            raise ValueError("Expected float32 [B,1,H,W] with B>=1 and H,W>=4")
        low = self.low_encoder(x)
        high = self.high_encoder(low)
        context, aspp_trace = self.aspp(high, return_trace=True)
        context_up = F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.low_projection(low)
        combined = torch.cat([context_up, detail], dim=1)
        decoded = self.decoder(combined)
        low_logits = self.head(decoded)
        logits = F.interpolate(low_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        trace = {"input": list(x.shape), "low_encoder": list(low.shape), "high_encoder": list(high.shape),
                 "aspp": aspp_trace, "context_up": list(context_up.shape), "low_projection": list(detail.shape),
                 "decoder_concatenation": list(combined.shape), "decoder": list(decoded.shape),
                 "low_logits": list(low_logits.shape), "logits": list(logits.shape)}
        return (logits, trace) if return_trace else logits


def smoke_step(seed=16):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        images, targets = next(iter(DataLoader(GroupedToyDataset([0]), batch_size=2, shuffle=False, num_workers=0)))
        before_images, before_targets = images.clone(), targets.clone()
        model = MiniDeepLabV3Plus().cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        criterion = nn.CrossEntropyLoss()
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        logits, trace = model(images, return_trace=True)
        validate_targets(logits, targets)
        loss = criterion(logits, targets)
        probability_before = logits.detach().softmax(1)[0, 1].numpy().copy()
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Missing or nonfinite gradient for {name}")
            gradients[name] = float(parameter.grad.norm())
        optimizer.step()
        changes = {name: float((p.detach() - before[name]).norm()) for name, p in model.named_parameters()}
        if not any(value > 0 for value in changes.values()):
            raise RuntimeError("Optimizer did not update parameters")
        with torch.no_grad():
            after = model(images)
            after_loss = float(criterion(after, targets))
            _, odd_trace = model(torch.zeros(1, 1, 65, 67), return_trace=True)
            ten_logits = MiniDeepLabV3Plus(10)(torch.zeros(1, 1, 48, 48))
        if not torch.equal(images, before_images) or not torch.equal(targets, before_targets):
            raise RuntimeError("Source tensors changed")
        blocks = ("low_encoder", "high_encoder", "aspp", "low_projection", "decoder", "head")
        report = {"seed": seed, "steps": 1, "optimizer": "Adam", "learning_rate": .005, "device": "cpu",
                  "parameters": sum(p.numel() for p in model.parameters()), "initial_loss": float(loss.detach()),
                  "same_batch_loss_after_step": after_loss, "trace": trace, "odd_size_trace": odd_trace,
                  "ten_class_shape": list(ten_logits.shape), "gradient_norms": gradients,
                  "block_gradient_norm_sums": {block: sum(value for name, value in gradients.items() if name.startswith(block + ".")) for block in blocks},
                  "parameter_update_norms": changes, "source_tensors_preserved": True,
                  "dilation_rates": [1, 2, 3], "effective_kernel_widths": [3, 5, 7],
                  "semantics": "Artificial bright background and dark objects. Original ten-class head retains geometric orientation IDs 0..8 and ferrite 9.",
                  "implementation": "Educational DeepLab-style reconstruction from written requirements. Referenced package unavailable. Standard ASPP convolutions, separable high encoder/decoder, no normalization layers or pretrained backbone."}
        arrays = {"image": images[0, 0].numpy(), "truth": targets[0].numpy(),
                  "before": probability_before, "after": after.softmax(1)[0, 1].numpy(),
                  "prediction": after.argmax(1)[0].numpy()}
    return report, arrays


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output directory exists. Choose a fresh directory")
    report, arrays = smoke_step()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    report["environment"] = {"torch": str(torch.__version__), "numpy": np.__version__, "cuda_available": torch.cuda.is_available()}
    report["code"] = {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                      "script_sha256": checksum(Path(__file__)),
                      "dataset_script_sha256": checksum(Path(__file__).with_name("session13_training_lab.py"))}
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), layout="constrained")
    try:
        for ax, (key, title) in zip(axes, (("image", "Artificial input"), ("truth", "Known mask"),
                                         ("before", "P(class 1) before"), ("after", "P(class 1) after"),
                                         ("prediction", "Argmax after one step"))):
            ax.imshow(arrays[key], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle("Session 16 MiniDeepLabV3+ artificial architecture smoke test")
        fig.savefig(output / "session16_architecture_smoke.png", dpi=120)
    finally:
        plt.close(fig)
    (output / "session16_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Parameters", report["parameters"], "loss before", report["initial_loss"], "loss after", report["same_batch_loss_after_step"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session16_v1")
    args = parser.parse_args()
    try:
        run(args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
