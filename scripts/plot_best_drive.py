#!/usr/bin/env python3
"""Plot the executed matched PPO time-penalty comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "output/best_drive/model/training.json").read_text())
    candidates = report["selection"]["ppo_time_penalty_candidates"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    colors = ("#777777", "#176a9a")
    labels = [f"−{item['time_penalty_per_second']:g} / second" for item in candidates]
    axes[0].bar(labels, [100 * item["closed_loop_validation"]["reach_rate"] for item in candidates], color=colors)
    axes[0].set(ylabel="Validation reach rate (%)", ylim=(0, 108), title="40 matched validation episodes")
    axes[1].bar(labels, [item["closed_loop_validation"]["mean_episode_seconds"] for item in candidates], color=colors)
    axes[1].set(ylabel="Mean episode duration (seconds)", title="Includes failed episode timeouts")
    for label, color in zip(("baseline", "requested"), colors):
        rows = [row for row in report["ppo"]["updates"] if row["penalty_label"] == label]
        cost = rows[0]["time_penalty_per_second"]
        axes[2].plot([row["update"] for row in rows],
                     [row["rollout_mean_success_seconds"] for row in rows],
                     color=color, label=f"−{cost:g} / second")
    axes[2].set(xlabel="PPO update", ylabel="Successful rollout duration (seconds)",
                title="24 stochastic episodes per update")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Stronger elapsed-time reward in native wheel-contact training", fontsize=14)
    figure.savefig(root / "output/best_design/drive_time_penalty_comparison.png", dpi=170)


if __name__ == "__main__":
    main()
