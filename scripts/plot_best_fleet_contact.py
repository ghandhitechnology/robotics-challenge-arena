#!/usr/bin/env python3
"""Plot the recorded G4 sample loss around the KIT/LAB contact."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "output/best_design/lab_transit_development/g4_1300600412/lab_contact_summary.json"
    report = json.loads(source.read_text())
    rows = report["loss_window"]
    times = [row["time_s"] for row in rows]
    collision = report["first_collision_state"]["time_s"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    axes[0].plot(times, [1000 * row["sample_xyz"][2] for row in rows], "o-", color="#236d88")
    axes[0].axhline(20, color="#999999", linestyle=":", label="Carry-height threshold")
    axes[0].set(ylabel="Sample center height (mm)", title="Sample falls after the contact")
    axes[1].plot(times, [1000 * row["sample_tool_distance_m"] for row in rows], "o-", color="#b15c35")
    axes[1].set(ylabel="Sample-to-tool distance (mm)", title="Grip loses the sample")
    for axis in axes:
        axis.axvline(collision, color="#333333", linestyle="--", linewidth=1,
                    label=f"KIT chute / LAB pad contact at {collision:.2f} s")
        axis.set(xlabel="Simulated time (seconds)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("G4 seed 1300600412: an unnecessary return crossing dislodged the sample", fontsize=13)
    figure.supxlabel("Native saved states at 20 ms intervals. No release command was issued.", fontsize=10)
    figure.savefig(root / "output/best_design/kit_lab_contact_diagnosis.png", dpi=170)


if __name__ == "__main__":
    main()
