"""Controlled binary augmentation and robustness comparison on artificial groups."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from session03_dataset_audit import ROOT, checksum
from session12_unet_architecture import SmallUNet
from session13_training_lab import (GroupedToyDataset, foreground_soft_dice_loss,
                                    is_better, restore_checkpoint, save_checkpoint,
                                    training_class_weights, validate)

CONDITIONS = ("nominal", "dim", "bright", "noisy")


def paired_transform(image, mask, rng, semantics="binary", geometry=True, photometric=True):
    if (not isinstance(image, np.ndarray) or image.ndim != 2 or not image.size
            or not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all()
            or np.any((image < 0) | (image > 1))):
        raise ValueError("Image must be finite floating 2D intensities in [0,1]")
    if semantics not in ("binary", "orientation"):
        raise ValueError("Specify binary or orientation semantics")
    maximum = 1 if semantics == "binary" else 9
    if (not isinstance(mask, np.ndarray) or mask.shape != image.shape
            or not np.issubdtype(mask.dtype, np.integer) or np.any((mask < 0) | (mask > maximum))):
        raise ValueError("Mask must match the image and declared class schema")
    if semantics == "orientation" and geometry:
        raise ValueError("Original orientation labels require validated angle remapping before geometric augmentation")
    image, mask = image.copy(), mask.copy()
    if geometry:
        turns = int(rng.integers(0, 4))
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if rng.random() < .5:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if rng.random() < .5:
            image, mask = np.flipud(image), np.flipud(mask)
    if photometric:
        contrast = rng.uniform(.85, 1.15)
        brightness = rng.uniform(.85, 1.15)
        image = ((image - .5) * contrast + .5) * brightness
        image += rng.normal(0, rng.uniform(0, .03), image.shape)
        image = np.clip(image, 0, 1)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(mask)


def normalization_statistics(training):
    total = squared = 0.
    pixels = 0
    for image, _ in training:
        if image.ndim != 3 or image.shape[0] != 1 or not torch.isfinite(image).all():
            raise ValueError("Normalization requires finite channel-first grayscale training images")
        values = image.double()
        total += float(values.sum())
        squared += float((values * values).sum())
        pixels += values.numel()
    if not pixels:
        raise ValueError("Training images are empty")
    mean = total / pixels
    std = float(np.sqrt(max(0., squared / pixels - mean**2)))
    if std <= 1e-8:
        raise ValueError("Training intensity variance is insufficient")
    return {"mean": mean, "std": std, "training_pixels": pixels}


def stress_image(image, condition, index):
    if condition not in CONDITIONS:
        raise ValueError("Unknown artificial stress condition")
    if (not isinstance(image, np.ndarray) or image.ndim != 2 or not image.size
            or not np.issubdtype(image.dtype, np.floating) or not np.isfinite(image).all()
            or np.any((image < 0) | (image > 1))):
        raise ValueError("Stress input must be finite floating 2D intensities in [0,1]")
    if condition == "dim":
        image = image * .5
    elif condition == "bright":
        image = image * 1.3
    elif condition == "noisy":
        image = image + np.random.default_rng(15000 + index).normal(0, .08, image.shape)
    return np.ascontiguousarray(np.clip(image, 0, 1), dtype=np.float32)


class NormalizedToyDataset(Dataset):
    def __init__(self, base, statistics, augmented=False, condition="nominal"):
        if (condition not in CONDITIONS or not np.isfinite(statistics["mean"])
                or not np.isfinite(statistics["std"]) or statistics["std"] <= 0):
            raise ValueError("Invalid stress condition or normalization statistics")
        if augmented and condition != "nominal":
            raise ValueError("Training augmentation and evaluation stress require separate datasets")
        self.base = base
        self.statistics = dict(statistics)
        self.augmented = augmented
        self.condition = condition
        self.epoch = 0

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        image, mask = self.base[index]
        image = image[0].numpy().copy()
        mask = mask.numpy().copy()
        if self.augmented:
            rng = np.random.default_rng(15 + self.epoch * 100003 + index)
            image, mask = paired_transform(image, mask, rng)
        image = stress_image(image, self.condition, index)
        image = (image - self.statistics["mean"]) / self.statistics["std"]
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0), torch.from_numpy(mask.copy()).long()


def state_checksum(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_condition(output, augmented, epochs, training, validation, statistics):
    name = "augmented" if augmented else "unaugmented"
    torch.manual_seed(15)
    model = SmallUNet().cpu()
    initial_hash = state_checksum(model)
    generator = torch.Generator().manual_seed(15)
    train_data = NormalizedToyDataset(training, statistics, augmented=augmented)
    val_data = NormalizedToyDataset(validation, statistics)
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=0)
    weights, counts = training_class_weights(training)
    optimizer = torch.optim.Adam(model.parameters(), lr=.015)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history, best = [], None
    path = output / f"session15_{name}_best_state.pt"
    config = {"seed": 15, "epochs": epochs, "batch_size": 4, "learning_rate": .015, "dice_weight": .25,
              "normalization": statistics, "augmented": augmented,
              "train_groups": list(training.groups), "validation_groups": list(validation.groups),
              "class_weights": weights.tolist()}
    for epoch in range(1, epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        loss_sum = 0.
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets) + .25 * foreground_soft_dice_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing or nonfinite training gradients")
            optimizer.step()
            loss_sum += float(loss.detach()) * images.shape[0]
        metrics = validate(model, val_loader)
        history.append({"epoch": epoch, "training_composite_loss": loss_sum / len(training), "validation": metrics})
        if is_better(metrics, best):
            best = metrics.copy()
            save_checkpoint(path, model, optimizer, epoch, metrics, config)
        print(name, epoch, "validation IoU", metrics["pooled_foreground_iou"])
    restored, checkpoint = restore_checkpoint(path)
    restored_metrics = validate(restored, val_loader)
    if restored_metrics != best:
        raise RuntimeError("Restored nominal validation metrics differ from selection")
    before = state_checksum(restored)
    stresses = {}
    for condition in CONDITIONS:
        dataset = NormalizedToyDataset(validation, statistics, condition=condition)
        stresses[condition] = validate(restored, DataLoader(dataset, batch_size=4, num_workers=0))
    if state_checksum(restored) != before:
        raise RuntimeError("Stress evaluation changed model parameters")
    images, _ = next(iter(val_loader))
    with torch.inference_mode():
        prediction = restored(images).argmax(1)[0].numpy().copy()
    return {"initial_state_sha256": initial_hash, "training_pixel_counts": counts.tolist(),
            "history": history, "best_epoch": checkpoint["epoch"], "selected_validation": best,
            "restoration_verified": True, "stresses": stresses, "checkpoint_sha256": checksum(path)}, prediction


def run(output, epochs=3):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Output directory exists. Choose a fresh directory")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Epochs must be a positive integer")
    training = GroupedToyDataset(range(8))
    validation = GroupedToyDataset(range(8, 12))
    statistics = normalization_statistics(training)
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            results, predictions = {}, {}
            for name, augmented in (("unaugmented", False), ("augmented", True)):
                results[name], predictions[name] = train_condition(output, augmented, epochs, training, validation, statistics)
            if results["unaugmented"]["initial_state_sha256"] != results["augmented"]["initial_state_sha256"]:
                raise RuntimeError("Comparison initial weights differ")
            report = {"normalization": statistics, "train_groups": list(training.groups),
                      "validation_groups": list(validation.groups), "epochs": epochs, "results": results,
                      "augmentation": {"rotation_quarter_turns": [0, 1, 2, 3], "horizontal_flip_probability": .5,
                                       "vertical_flip_probability": .5, "brightness": [.85, 1.15],
                                       "contrast_about_half_intensity": [.85, 1.15], "noise_sigma": [0, .03],
                                       "seed_formula": "15 + epoch * 100003 + sample_index"},
                      "stress_parameters": {"dim_multiplier": .5, "bright_multiplier": 1.3, "noise_sigma": .08,
                                            "noise_seed_formula": "15000 + sample_index"},
                      "selection_rule": "Nominal pooled validation foreground IoU, then lower unweighted CE, retaining earlier exact ties",
                      "scope": "Two artificial validation-selected development runs. Reconstructed Session 13 generator and Session 15 implementation.",
                      "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "cpu_threads": 1}}
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
            report["code"] = {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                              "script_sha256": checksum(Path(__file__)),
                              "dataset_script_sha256": checksum(Path(__file__).with_name("session13_training_lab.py"))}
            image, mask = training[0]
            example_image, example_mask = paired_transform(image[0].numpy(), mask.numpy(), np.random.default_rng(15))
            fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
            try:
                for name in results:
                    history = results[name]["history"]
                    axes[0, 0].plot([row["epoch"] for row in history], [row["validation"]["pooled_foreground_iou"] for row in history], marker="o", label=name)
                axes[0, 0].set(title="Nominal validation IoU", xlabel="Epoch", ylim=(0, 1))
                axes[0, 0].legend()
                positions = np.arange(len(CONDITIONS))
                for offset, name in ((-.18, "unaugmented"), (.18, "augmented")):
                    axes[0, 1].bar(positions + offset, [results[name]["stresses"][c]["pooled_foreground_iou"] for c in CONDITIONS], width=.36, label=name)
                axes[0, 1].set(title="Selected-checkpoint stress IoU", xticks=positions, xticklabels=CONDITIONS, ylim=(0, 1))
                axes[0, 1].legend(fontsize=8)
                for ax, pixels, title in (
                    (axes[0, 2], image[0].numpy(), "Original training image"), (axes[0, 3], mask.numpy(), "Original mask"),
                    (axes[1, 0], example_image, "Augmented training image"), (axes[1, 1], example_mask, "Paired augmented mask"),
                    (axes[1, 2], predictions["unaugmented"], "Unaugmented model validation mask"),
                    (axes[1, 3], predictions["augmented"], "Augmented model validation mask")):
                    ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ax.set_title(title)
                    ax.set_axis_off()
                fig.suptitle("Session 15 artificial augmentation and robustness comparison")
                fig.savefig(output / "session15_robustness.png", dpi=120)
            finally:
                plt.close(fig)
            (output / "session15_robustness.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            return report
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session15_v1")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args.output_dir, args.epochs)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
