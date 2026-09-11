#!/usr/bin/env python3
"""Plot measured fleet tuning and the phases of a saved native mission."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def category(phase):
    if phase.startswith("wait"):
        return "Waiting"
    if phase == "finished":
        return "Finished"
    if phase.startswith("pick"):
        return "Pickup"
    if phase.startswith(("deliver", "drop", "unload")):
        return "Delivery"
    return "Travel / deployment"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tuning = json.loads((args.search / "tune.json").read_text())
    mission = json.loads((args.mission / "report.json").read_text())
    candidates = tuning["candidates"]
    labels = [item["name"].replace("uniform_", "All ").replace("fast_couriers_", "Couriers ").replace("_", " / ") for item in candidates]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    positions = np.arange(len(candidates))
    colors = ["#27789b" if item["lab_limits"] is not None else "#929ca3" for item in candidates]
    axes[0].barh(positions, [100 * item["full_success_rate"] for item in candidates], color=colors)
    axes[0].set(yticks=positions, yticklabels=labels, xlim=(0, 110), xlabel="Full-score success rate (%)",
                title="Matched validation physics")
    axes[0].invert_yaxis()
    for position, candidate in enumerate(candidates):
        axes[0].text(101, position, f"{round(candidate['full_success_rate'] * candidate['episodes'])}/{candidate['episodes']}", va="center", fontsize=9)
    means = [item["mean_success_seconds"] if item["mean_success_seconds"] is not None else 0 for item in candidates]
    axes[1].barh(positions, means, color=colors)
    axes[1].set(yticks=positions, yticklabels=labels, xlim=(0, 130), xlabel="Mean successful mission time (s)",
                title="Timing excludes failed missions")
    axes[1].invert_yaxis()
    axes[1].axvline(120, color="#bc4940", linestyle="--", linewidth=1)
    for position, seconds in enumerate(means):
        axes[1].text(seconds + 2, position, f"{seconds:.1f}" if seconds else "No successful run", va="center", fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=.15)
        axis.set_axisbelow(True)
    figure.suptitle("Five-robot speed selection: complete every delivery before saving time", fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "fleet_speed_comparison.png", dpi=170)
    plt.close(figure)

    palette = {"Waiting": "#d6d9dc", "Finished": "#f1f2f3", "Pickup": "#ae762c",
               "Delivery": "#247b72", "Travel / deployment": "#526f92"}
    figure, axis = plt.subplots(figsize=(13, 4.4), layout="constrained")
    roles = ["lab", "red", "yellow", "kit", "green"]
    trace = mission["traffic_trace"]
    for role_index, role in enumerate(roles):
        for index, state in enumerate(trace):
            end = trace[index + 1]["time_s"] if index + 1 < len(trace) else mission["declaration_seconds"]
            phase = category(state["robots"][role]["phase"])
            axis.broken_barh([(state["time_s"], max(0., end - state["time_s"]))],
                            (role_index - .32, .64), facecolors=palette[phase])
    axis.axvline(mission["declaration_seconds"], color="#243c51", linewidth=1.4)
    axis.axvspan(mission["declaration_seconds"], mission["declaration_seconds"] + 5, color="#d0e5db")
    axis.set(yticks=range(5), yticklabels=[name.upper() for name in roles],
             xlabel="Simulated time from start (seconds)", xlim=(0, max(90, mission["declaration_seconds"] + 7)),
             title=f"Native mission: {mission['score_at_declaration']['score']} points at {mission['declaration_seconds']:.2f} s, then five-second hold")
    axis.invert_yaxis()
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.grid(axis="x", alpha=.15)
    axis.legend(handles=[Patch(facecolor=color, label=label) for label, color in palette.items()],
                ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16))
    figure.savefig(args.output / "fleet_mission_timeline.png", dpi=170)


if __name__ == "__main__":
    main()
