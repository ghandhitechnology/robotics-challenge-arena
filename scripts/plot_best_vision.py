#!/usr/bin/env python3
"""Plot the matched G4 CNN training histories without reading the test split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/best_vision"))
    args = parser.parse_args()
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for folder, label, color in (("control", "Control, learning rate 0.003", "#777777"),
                                 ("refined", "Refined, learning rate 0.001", "#176a9a")):
        rows = [json.loads(line) for line in (args.root / folder / "history.jsonl").read_text().splitlines()]
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [100 * row["validation_accuracy"] for row in rows], label=label, color=color)
        axes[1].plot(epochs, [row["validation_center_mean_pixels"] for row in rows], label=label, color=color)
    axes[0].set(ylabel="Validation classification accuracy (%)", ylim=(95, 100.05))
    axes[1].set(ylabel="Validation mean center error (pixels)", ylim=(0, 3))
    for axis in axes:
        axis.set(xlabel="Training epoch")
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    figure.suptitle("CNN refinement on the same frozen MuJoCo image dataset", fontsize=14)
    figure.savefig(args.root / "training_comparison.png", dpi=170)


if __name__ == "__main__":
    main()
