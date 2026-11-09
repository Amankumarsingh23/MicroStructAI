"""Controlled artificial architecture comparison with restored-state diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from session03_dataset_audit import ROOT, checksum
from session12_unet_architecture import SmallUNet
from session13_training_lab import (GroupedToyDataset, foreground_soft_dice_loss, is_better,
                                    save_checkpoint, training_class_weights, validate)
from session14_evaluation_lab import infer_scenes, scene_metrics, summarize, thin_diagnostic
from session15_augmentation_robustness import NormalizedToyDataset, normalization_statistics
from session16_deeplab_architecture import MiniDeepLabV3Plus

ARCHITECTURES = ("unet", "mini_deeplab")


def build_model(name):
    if name == "unet":
        return SmallUNet(classes=2)
    if name == "mini_deeplab":
        return MiniDeepLabV3Plus(classes=2)
    raise ValueError("Unknown comparison architecture")


def restore_model(path, architecture):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["architecture"] != architecture:
        raise ValueError("Checkpoint architecture differs from requested model")
    model = build_model(architecture).cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def cpu_latency(model, sample, repetitions=15, warmup=5):
    if repetitions < 1 or warmup < 0 or sample.device.type != "cpu" or sample.shape[0] != 1:
        raise ValueError("Latency requires positive repetitions and one CPU input image")
    model.eval()
    elapsed = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            model(sample)
            elapsed.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms_per_image": float(np.median(elapsed)), "minimum_ms": min(elapsed),
            "maximum_ms": max(elapsed), "repetitions": repetitions, "warmup": warmup,
            "input_shape": list(sample.shape), "dtype": str(sample.dtype),
            "cpu_threads": torch.get_num_threads(), "scope": "Warmed model forward pass with preloaded normalized input"}


def train_one(output, architecture, seed, epochs, training, validation, statistics):
    torch.manual_seed(seed)
    model = build_model(architecture).cpu()
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=4, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(validation, batch_size=4, shuffle=False, num_workers=0)
    weights, counts = training_class_weights(training.base)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    criterion = nn.CrossEntropyLoss(weight=weights)
    config = {"architecture": architecture, "seed": seed, "epochs": epochs,
              "normalization": statistics, "class_weights": weights.tolist(),
              "train_groups": list(training.base.groups), "validation_groups": list(validation.base.groups),
              "batch_size": 4, "learning_rate": .005, "dice_weight": .25,
              "augmentation": False, "cpu_threads": 1, "num_workers": 0}
    path = output / f"session17_{architecture}_seed{seed}_best_state.pt"
    best, history = None, []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        for images, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets) + .25 * foreground_soft_dice_loss(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing or nonfinite training gradient")
            optimizer.step()
            total_loss += float(loss.detach()) * images.shape[0]
        metrics = validate(model, val_loader)
        history.append({"architecture": architecture, "seed": seed, "epoch": epoch,
                        "training_composite_loss": total_loss / len(training), **metrics})
        if is_better(metrics, best):
            best = metrics.copy()
            save_checkpoint(path, model, optimizer, epoch, metrics, config)
        print(architecture, seed, epoch, "validation IoU", metrics["pooled_foreground_iou"])
    train_seconds = time.perf_counter() - start
    restored, checkpoint = restore_model(path, architecture)
    if validate(restored, val_loader) != best:
        raise RuntimeError("Restored checkpoint does not reproduce selected metrics")
    scenes = infer_scenes(restored, validation)
    method = f"{architecture}_seed{seed}"
    rows = [{"scene_id": scene["scene_id"], "group_id": scene["group_id"], "method": method,
             **scene_metrics(scene["truth"], (scene["probability"] > .5).astype(np.uint8),
                             np.ones(scene["truth"].shape, bool))} for scene in scenes]
    evaluation = summarize(rows)[method]
    if not np.isclose(evaluation["per_class"]["1"]["iou"], best["pooled_foreground_iou"], rtol=0, atol=1e-8):
        raise RuntimeError("Detailed evaluation disagrees with checkpoint validation")
    probe_truth, _, thin = thin_diagnostic()
    probe_image = np.where(probe_truth, 55., 205.) / 255
    probe_image = (probe_image - statistics["mean"]) / statistics["std"]
    probe_tensor = torch.from_numpy(probe_image.astype(np.float32))[None, None]
    with torch.inference_mode():
        probe_prediction = restored(probe_tensor).argmax(1)[0].numpy()
    probe_metrics = scene_metrics(probe_truth, probe_prediction, np.ones(probe_truth.shape, bool), thin)
    sample = validation[0][0].unsqueeze(0)
    latency = cpu_latency(restored, sample)
    parameter_bytes = sum(p.numel() * p.element_size() for p in restored.parameters())
    return {"architecture": architecture, "seed": seed, "parameters": sum(p.numel() for p in restored.parameters()),
            "parameter_storage_bytes": parameter_bytes, "training_pixel_counts": counts.tolist(),
            "training_wall_seconds_including_validation_and_checkpoint_writes": train_seconds,
            "history": history, "best_epoch": checkpoint["epoch"], "selected_validation": best,
            "restoration_verified": True, "checkpoint_sha256": checksum(path),
            "evaluation": evaluation, "per_scene": rows, "thin_probe": probe_metrics,
            "cpu_latency": latency}, (scenes[0]["probability"] > .5).astype(np.uint8)


def run(output, epochs=4, seeds=(17, 18)):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Comparison directory exists. Choose a fresh directory")
    seeds = tuple(seeds)
    if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1 or not seeds
            or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
        raise ValueError("Provide positive epochs and distinct nonnegative integer seeds")
    train_base, val_base = GroupedToyDataset(range(8)), GroupedToyDataset(range(8, 12))
    if set(train_base.groups) & set(val_base.groups):
        raise ValueError("Training and validation groups overlap")
    statistics = normalization_statistics(train_base)
    training = NormalizedToyDataset(train_base, statistics)
    validation = NormalizedToyDataset(val_base, statistics)
    output.mkdir(parents=True, exist_ok=False)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            results, predictions = [], {}
            for architecture in ARCHITECTURES:
                for seed in seeds:
                    result, prediction = train_one(output, architecture, seed, epochs, training, validation, statistics)
                    results.append(result)
                    if seed == seeds[0]:
                        predictions[architecture] = prediction
            baseline_rows = []
            for index in range(len(val_base)):
                image, target = val_base[index]
                truth = target.numpy()
                prediction = (np.rint(image[0].numpy() * 255) <= 120).astype(np.uint8)
                baseline_rows.append({"scene_id": f"group-{val_base.groups[index // 2]}-view-{index % 2}",
                                      "group_id": val_base.groups[index // 2], "method": "Fixed 120",
                                      **scene_metrics(truth, prediction, np.ones(truth.shape, bool))})
            baseline = summarize(baseline_rows)["Fixed 120"]
            seed_summary = {}
            for architecture in ARCHITECTURES:
                values = [r["selected_validation"]["pooled_foreground_iou"] for r in results if r["architecture"] == architecture]
                seed_summary[architecture] = {"mean_selected_iou": float(np.mean(values)),
                                              "sample_standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                              "minimum_iou": min(values), "maximum_iou": max(values)}
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
            status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
            report = {"epochs": epochs, "seeds": list(seeds), "normalization": statistics,
                      "train_groups": list(train_base.groups), "validation_groups": list(val_base.groups),
                      "protocol": {"optimizer": "Adam", "learning_rate": .005, "dice_weight": .25, "batch_size": 4,
                                   "augmentation": False, "cpu_threads": 1, "class_weight_cap": 8,
                                   "selection": "Highest pooled validation foreground IoU, lower unweighted CE, earlier exact tie"},
                      "runs": results, "seed_summary": seed_summary, "fixed_120_baseline": baseline,
                      "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                                      "python": platform.python_version(), "platform": platform.platform(),
                                      "processor": platform.processor()},
                      "code": {"commit": git.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
                               "script_sha256": checksum(Path(__file__)),
                               "dataset_script_sha256": checksum(Path(__file__).with_name("session13_training_lab.py"))},
                      "scope": "Reconstructed artificial controlled comparison using existing educational networks and validation-selected checkpoints",
                      "memory_policy": "Parameter tensor storage only. Peak working memory and FLOPs require additional measurement.",
                      "latency_policy": "Five warmups and 15 timed CPU forwards, batch one, float32, one thread. Loading, preprocessing, postprocessing and transfers excluded.",
                      "uncertainty_policy": "Seed summaries are descriptive. Two seeds and four artificial validation groups provide limited variability evidence."}
            epoch_rows = [row for result in results for row in result["history"]]
            with (output / "session17_epochs.csv").open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(epoch_rows[0]))
                writer.writeheader()
                writer.writerows(epoch_rows)
            fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
            try:
                for result in results:
                    axes[0, 0].plot([r["epoch"] for r in result["history"]],
                                    [r["pooled_foreground_iou"] for r in result["history"]],
                                    marker="o", label=f"{result['architecture']} seed {result['seed']}")
                axes[0, 0].set(title="Validation trajectories", xlabel="Epoch", ylabel="Pooled foreground IoU", ylim=(0, 1))
                axes[0, 0].legend(fontsize=7)
                names = [f"{r['architecture']}\nseed {r['seed']}" for r in results]
                axes[0, 1].bar(names + ["Fixed 120"], [r["selected_validation"]["pooled_foreground_iou"] for r in results] + [baseline["per_class"]["1"]["iou"]])
                axes[0, 1].set(title="Selected nominal IoU", ylim=(0, 1))
                axes[0, 1].tick_params(axis="x", labelsize=6)
                axes[0, 2].bar(names, [r["thin_probe"]["thin_feature_recall"] for r in results])
                axes[0, 2].set(title="Constructed thin-probe recall", ylim=(0, 1))
                axes[0, 2].tick_params(axis="x", labelsize=6)
                for ax, pixels, title in ((axes[1, 0], val_base[0][1].numpy(), "Known validation mask"),
                                          (axes[1, 1], predictions["unet"], f"U-Net seed {seeds[0]}"),
                                          (axes[1, 2], predictions["mini_deeplab"], f"MiniDeepLab seed {seeds[0]}")):
                    ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                    ax.set_title(title)
                    ax.set_axis_off()
                fig.suptitle("Session 17 controlled artificial architecture comparison")
                fig.savefig(output / "session17_comparison.png", dpi=120)
            finally:
                plt.close(fig)
            (output / "session17_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            return report
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/session17_v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    args = parser.parse_args()
    try:
        run(args.output_dir, args.epochs, args.seeds)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
