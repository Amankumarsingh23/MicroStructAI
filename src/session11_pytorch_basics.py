"""One CPU SGD step on artificial data; not a trained segmentation benchmark."""
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
from torch.utils.data import DataLoader, Dataset

from session03_dataset_audit import ROOT, checksum


def binary_from_orientation(original_mask):
    if (not isinstance(original_mask, np.ndarray) or original_mask.ndim != 2 or not original_mask.size
            or not np.issubdtype(original_mask.dtype, np.integer)
            or np.any((original_mask < 0) | (original_mask > 9))):
        raise ValueError("Expected a nonempty 2D integer orientation mask with IDs 0..9")
    return np.where(original_mask == 9, 0, 1).astype(np.uint8)


def toy_pair(index):
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("Toy index must be a nonnegative integer")
    rows, cols = np.indices((64, 64))
    mask = (((rows - (29 + index % 3)) ** 2 + (cols - 26) ** 2 <= 12**2)
            | ((rows - 48)**2 + (cols - (49 - index % 3))**2 <= 4**2)).astype(np.uint8)
    noise = np.random.default_rng(1100 + index).normal(0, 6, mask.shape)
    gray = np.clip(np.rint(np.where(mask, 55, 205) + noise), 0, 255).astype(np.uint8)
    return gray, mask


class ToyMicrographDataset(Dataset):
    def __init__(self, count=8):
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Dataset count must be a positive integer")
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError("Toy sample index out of range")
        gray, mask = toy_pair(index)
        image = torch.from_numpy(gray.copy()).float().unsqueeze(0) / 255.0
        target = torch.from_numpy(mask.copy()).long()
        return image, target


class TinyPixelNet(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        if not isinstance(classes, int) or isinstance(classes, bool) or classes < 2:
            raise ValueError("Class count must be an integer >=2")
        self.layers = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, classes, 1))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or images.dtype != torch.float32:
            raise ValueError("Expected float32 [B,1,H,W] input")
        return self.layers(images)


def validate_targets(logits, masks):
    if (logits.ndim != 4 or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or masks.dtype != torch.long or masks.shape != (logits.shape[0], *logits.shape[2:])
            or not masks.numel() or torch.any(masks < 0) or torch.any(masks >= logits.shape[1])):
        raise ValueError("Cross-entropy requires finite [B,C,H,W] logits and long [B,H,W] IDs in 0..C-1")


def one_step(seed=17):
    # Local RNG context avoids changing the caller's global CPU random sequence.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        loader = DataLoader(ToyMicrographDataset(), batch_size=2, shuffle=False, num_workers=0)
        images, masks = next(iter(loader))
        model = TinyPixelNet().cpu()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=.2)
        original_images, original_masks = images.clone(), masks.clone()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer.zero_grad()
        logits = model(images)
        validate_targets(logits, masks)
        loss = criterion(logits, masks)  # Raw logits; no softmax here.
        loss.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Missing or nonfinite gradient")
            gradients[name] = float(parameter.grad.norm())
        optimizer.step()
        changes = {name: float((parameter.detach() - before[name]).norm()) for name, parameter in model.named_parameters()}
        if not any(value > 0 for value in changes.values()):
            raise RuntimeError("Optimizer did not change parameters")
        with torch.no_grad():
            after_logits = model(images)
            after_loss = criterion(after_logits, masks)
            probabilities = after_logits.softmax(dim=1)
            predictions = after_logits.argmax(dim=1)
        assert torch.equal(images, original_images) and torch.equal(masks, original_masks)
        report = {"seed": seed, "device": "cpu", "steps": 1, "learning_rate": .2,
                  "dataset_count": 8, "batch_size": 2, "shuffle": False, "num_workers": 0,
                  "image_shape": list(images.shape), "image_dtype": str(images.dtype),
                  "target_shape": list(masks.shape), "target_dtype": str(masks.dtype),
                  "feature_shape": list(model.layers[0](images).shape), "logits_shape": list(logits.shape),
                  "prediction_shape": list(predictions.shape), "parameters": sum(p.numel() for p in model.parameters()),
                  "initial_loss": float(loss.detach()), "same_batch_loss_after_step": float(after_loss),
                  "gradient_norms": gradients, "sum_gradient_norms": sum(gradients.values()),
                  "parameter_update_norms": changes, "parameters_changed": any(v > 0 for v in changes.values()),
                  "class_semantics": "Toy 0=bright background, 1=artificial dark object; not verified material labels",
                  "scope": "One optimizer step on first artificial batch; no validation/test performance or real-data access",
                  "fixture_note": "Reconstructed fixture; referenced source package not attached"}
        arrays = {"image": images[0, 0].numpy(), "target": masks[0].numpy(),
                  "probability": probabilities[0, 1].numpy(), "prediction": predictions[0].numpy(),
                  "kernel": model.layers[0].weight[0, 0].detach().numpy().copy()}
    return report, arrays


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output directory exists; choose a fresh output directory")
    report, arrays = one_step()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    report["environment"] = {"torch": torch.__version__, "numpy": np.__version__,
                             "python": platform.python_version(), "cuda_available": torch.cuda.is_available()}
    report["code"] = {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                       "script_sha256": checksum(Path(__file__))}
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    try:
        for ax, key, title in ((axes[0, 0], "image", "Artificial input"),
                               (axes[0, 1], "target", "Known artificial mask"),
                               (axes[0, 2], "probability", "Dark-object probability (after 1 step)"),
                               (axes[1, 0], "prediction", "Predicted class (after 1 step)")):
            ax.imshow(arrays[key], cmap="viridis" if key == "probability" else "gray",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        kernel = arrays["kernel"]
        limit = float(np.max(np.abs(kernel)))
        axes[1, 1].imshow(kernel, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
        axes[1, 1].set_title("Updated first 3x3 convolution filter")
        axes[1, 1].set_axis_off()
        axes[1, 2].text(.05, .5, "ONE CPU SGD STEP\nNo trained benchmark\nNo real micrographs\nNo test-set access", va="center")
        axes[1, 2].set_axis_off()
        fig.suptitle("Session 11 | Toy tensors, logits and gradients")
        fig.savefig(output / "session11_tensor_lab.png", dpi=120)
    finally:
        plt.close(fig)
    (output / "session11_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session11")
    args = parser.parse_args()
    try:
        run(args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
