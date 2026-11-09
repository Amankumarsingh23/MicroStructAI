"""Artificial grouped U-Net training/validation, with best-state restoration."""
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
from session11_pytorch_basics import validate_targets
from session12_unet_architecture import SmallUNet


class GroupedToyDataset(Dataset):
    """Each generated geometry group has even/shaded views, always kept together."""
    def __init__(self, groups):
        self.groups = tuple(groups)
        if (not self.groups or len(set(self.groups)) != len(self.groups)
                or any(not isinstance(g, int) or g < 0 for g in self.groups)):
            raise ValueError("Provide distinct nonnegative artificial group IDs")

    def __len__(self):
        return 2 * len(self.groups)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Artificial sample index out of range")
        group, variant = self.groups[index // 2], index % 2
        rng = np.random.default_rng(1300 + group)
        r, c = np.indices((48, 48))
        center_r, center_c = rng.integers(15, 34, 2)
        radius = int(rng.integers(6, 10))
        mask = ((r - center_r)**2 + (c - center_c)**2 <= radius**2).astype(np.int64)
        clean = np.where(mask, 55., 205.)
        field = np.linspace(.75, 1.15, 48)[None, :] if variant else 1
        gray = np.clip(np.rint(clean * field + rng.normal(0, 5, mask.shape)), 0, 255).astype(np.uint8)
        return torch.from_numpy(gray.copy()).float().unsqueeze(0) / 255, torch.from_numpy(mask.copy()).long()


def training_class_weights(dataset):
    counts = torch.zeros(2, dtype=torch.long)
    for _, target in dataset:
        if target.dtype != torch.long or torch.any((target < 0) | (target > 1)):
            raise ValueError("Training weights require binary long targets")
        counts += torch.bincount(target.flatten(), minlength=2)
    if torch.any(counts == 0):
        raise ValueError("Both classes must occur in this training exercise")
    return torch.tensor([1., min(8., float(counts[0] / counts[1]))]), counts


def foreground_soft_dice_loss(logits, targets, eps=1e-6):
    validate_targets(logits, targets)
    if logits.shape[1] != 2 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Foreground Dice requires two-class logits and positive finite eps")
    probability = logits.softmax(dim=1)[:, 1]
    foreground = (targets == 1).float()
    intersection = (probability * foreground).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + foreground.sum(dim=(1, 2))
    return 1 - ((2 * intersection + eps) / (denominator + eps)).mean()


def validate(model, loader):
    """Pixel-weighted unweighted CE; pooled foreground overlap and scene-macro IoU."""
    was_training = model.training
    model.eval()
    tp = fp = fn = tn = pixels = 0
    ce_sum = 0.
    scene_ious = []
    try:
        with torch.no_grad():
            for images, targets in loader:
                logits = model(images)
                validate_targets(logits, targets)
                ce_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
                pixels += targets.numel()
                predictions = logits.argmax(1)
                tp += int(((predictions == 1) & (targets == 1)).sum())
                fp += int(((predictions == 1) & (targets == 0)).sum())
                fn += int(((predictions == 0) & (targets == 1)).sum())
                tn += int(((predictions == 0) & (targets == 0)).sum())
                for prediction, target in zip(predictions, targets):
                    intersection = int(((prediction == 1) & (target == 1)).sum())
                    union = int(((prediction == 1) | (target == 1)).sum())
                    if union:
                        scene_ious.append(intersection / union)
    finally:
        model.train(was_training)
    if not pixels:
        raise ValueError("Validation loader is empty")
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "unweighted_ce": ce_sum / pixels, "pooled_foreground_iou": tp / union if union else None,
            "pooled_foreground_dice": 2 * tp / (2 * tp + fp + fn) if union else None,
            "pixel_accuracy": (tp + tn) / pixels,
            "scene_macro_foreground_iou": float(np.mean(scene_ious)) if scene_ious else None}


def is_better(candidate, best):
    if best is None:
        return True
    candidate_iou = candidate["pooled_foreground_iou"]
    best_iou = best["pooled_foreground_iou"]
    a = candidate_iou if candidate_iou is not None else -1.
    b = best_iou if best_iou is not None else -1.
    return a > b or (a == b and candidate["unweighted_ce"] < best["unweighted_ce"])


def save_checkpoint(path, model, optimizer, epoch, metrics, config):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "validation": metrics, "config": config}, temporary)
    temporary.replace(path)


def restore_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SmallUNet(classes=2).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def run(output, epochs=5, dice_weight=.25):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Training output exists; choose a fresh directory")
    if not isinstance(epochs, int) or epochs < 1 or not np.isfinite(dice_weight) or dice_weight < 0:
        raise ValueError("Epochs must be positive and Dice weight finite/nonnegative")
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            training = GroupedToyDataset(range(8))
            validation = GroupedToyDataset(range(8, 12))
            assert not set(training.groups) & set(validation.groups)
            generator = torch.Generator().manual_seed(13)
            train_loader = DataLoader(training, batch_size=4, shuffle=True, generator=generator, num_workers=0)
            val_loader = DataLoader(validation, batch_size=4, shuffle=False, num_workers=0)
            weights, counts = training_class_weights(training)
            model = SmallUNet(classes=2).cpu()
            optimizer = torch.optim.Adam(model.parameters(), lr=.015)
            criterion = nn.CrossEntropyLoss(weight=weights)
            config = {"seed": 13, "epochs": epochs, "batch_size": 4, "learning_rate": .015,
                      "dice_weight": dice_weight, "train_groups": list(training.groups),
                      "validation_groups": list(validation.groups), "class_weights": weights.tolist(),
                      "num_workers": 0, "device": "cpu", "cpu_threads": 1}
            history, best, best_epoch = [], None, None
            checkpoint_path = output / "session13_best_state.pt"
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.
                sample_count = 0
                for images, targets in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(images)
                    validate_targets(logits, targets)
                    loss = criterion(logits, targets) + dice_weight * foreground_soft_dice_loss(logits, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError("Missing/nonfinite training gradient")
                    optimizer.step()
                    total_loss += float(loss.detach()) * images.shape[0]
                    sample_count += images.shape[0]
                metrics = validate(model, val_loader)
                history.append({"epoch": epoch, "training_composite_loss": total_loss / sample_count,
                                "validation": metrics})
                if is_better(metrics, best):
                    best, best_epoch = metrics.copy(), epoch
                    save_checkpoint(checkpoint_path, model, optimizer, epoch, metrics, config)
                print(f"Epoch {epoch}: train={total_loss/sample_count:.6f}, validation IoU={metrics['pooled_foreground_iou']}, CE={metrics['unweighted_ce']:.6f}")
            restored, checkpoint = restore_checkpoint(checkpoint_path)
            restored_metrics = validate(restored, val_loader)
            for key, value in best.items():
                if value is None:
                    assert restored_metrics[key] is None
                else:
                    assert np.isclose(value, restored_metrics[key], rtol=0, atol=1e-8), key
            assert checkpoint["epoch"] == best_epoch
            images, targets = next(iter(val_loader))
            with torch.no_grad():
                logits = restored(images)
                probabilities = logits.softmax(1)
                prediction = logits.argmax(1)
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
            report = {"config": config, "training_pixel_counts": counts.tolist(),
                      "training_class_fractions": (counts.double() / counts.sum()).tolist(),
                      "history": history, "best_epoch": best_epoch, "selected_validation": best,
                      "restored_validation": restored_metrics, "restoration_verified": True,
                      "checkpoint_sha256": checksum(checkpoint_path),
                      "selection_rule": "Pooled validation foreground IoU descending; unweighted validation CE ascending; exact ties keep earlier epoch",
                      "loss_policy": "Training WCE + coefficient * mean per-image foreground soft Dice; validation unweighted pixel-mean CE. Magnitudes are not comparable losses.",
                      "aggregation": "Pooled overlap uses total TP/FP/FN; scene macro excludes both-empty foreground. Undefined overlap is null.",
                      "environment": {"torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version()},
                      "code": {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()), "script_sha256": checksum(Path(__file__))},
                      "scope": "Reconstructed artificial training exercise; package not attached; no official-test access or real steel score",
                      "resume_limit": "Optimizer state saved; no resume command or exact RNG-state continuation is implemented"}
            fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
            try:
                epoch_numbers = [row["epoch"] for row in history]
                for ax, values, title in (
                    (axes[0, 0], [row["training_composite_loss"] for row in history], "Training composite loss"),
                    (axes[0, 1], [row["validation"]["unweighted_ce"] for row in history], "Validation unweighted CE"),
                    (axes[0, 2], [row["validation"]["pooled_foreground_iou"] for row in history], "Validation pooled foreground IoU")):
                    ax.plot(epoch_numbers, values, marker="o")
                    ax.set(title=title, xlabel="Epoch")
                axes[0, 3].text(0, .5, "Selection: validation IoU\nTie-break: validation CE\nOfficial tests: not accessed\nArtificial validation only", va="center")
                axes[0, 3].set_axis_off()
                for ax, values, title in (
                    (axes[1, 0], images[0, 0], "Artificial input"), (axes[1, 1], targets[0], "Known mask"),
                    (axes[1, 2], probabilities[0, 1], "Best-checkpoint P(class 1)"),
                    (axes[1, 3], prediction[0], "Best-checkpoint argmax")):
                    ax.imshow(values.numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ax.set_title(title)
                    ax.set_axis_off()
                fig.suptitle("Artificial U-Net training: validation selection, no test evaluation")
                fig.savefig(output / "session13_training.png", dpi=120)
            finally:
                plt.close(fig)
            (output / "session13_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            return report
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session13_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dice-weight", type=float, default=.25)
    args = parser.parse_args()
    try:
        run(args.output_dir, args.epochs, args.dice_weight)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
