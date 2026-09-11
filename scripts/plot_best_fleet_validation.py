#!/usr/bin/env python3
"""Plot the complete frozen fleet evaluation and preserve its source digest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.test.read_text())
    rows = report["episodes"]
    if len(report["candidates"]) != 1 or len(rows) != len(report["seeds"]):
        raise ValueError("Expected one frozen profile and a complete evaluation")
    candidate = report["candidates"][0]
    successes = sum(row["success"] for row in rows)
    times = np.array([row["declaration_seconds"] for row in rows if row["success"]])
    if not len(times):
        raise ValueError("No successful mission timing to plot")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [.9, 1.2]})
    fig.patch.set_facecolor("#faf8f3")
    axes[0].axis("off")
    axes[0].text(0, .85, f"{successes}/{len(rows)}", fontsize=48, fontweight="bold", color="#1c7c82")
    axes[0].text(0, .70, "Complete 160-point missions", fontsize=16)
    details = (
        f"{100 * candidate['wilson_lower_95']:.2f}% Wilson 95% lower bound\n\n"
        f"{candidate['mean_success_seconds']:.2f} s mean successful time\n"
        f"{candidate['p95_success_seconds']:.2f} s successful time p95\n\n"
        f"{len(rows) - successes} failed missions retained\n"
        f"{candidate['exceptions']} numerical exceptions"
    )
    axes[0].text(0, .09, details, fontsize=13, linespacing=1.45, color="#29343b")
    ax = axes[1]
    ax.set_facecolor("#faf8f3")
    ax.spines[["top", "right"]].set_visible(False)
    ax.hist(times, bins=18, color="#1c7c82", edgecolor="#faf8f3")
    ax.axvline(120, color="#b05234", linestyle="--", linewidth=1.6, label="120 s limit")
    ax.axvline(float(np.quantile(times, .95)), color="#29343b", linestyle=":", label="Successful time p95")
    ax.set(xlabel="Declaration time (simulated seconds)", ylabel="Successful missions")
    ax.legend(frameon=False)
    fig.suptitle("Five-robot fleet: frozen G4 evaluation", x=.05, ha="left", fontsize=18, fontweight="bold")
    fig.text(.05, .025,
             f"Native MuJoCo, source {report['commit'][:7]}. All 16 releases, stable declaration and five-second hold required.\n"
             "Physical parameters vary; field marks and starting poses stay fixed. Hardware qualification is unmeasured.",
             fontsize=9, color="#49555e")
    fig.tight_layout(rect=(.025, .12, .99, .91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    manifest = {"test": str(args.test), "test_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
                "source_commit": report["commit"], "episodes": len(rows), "successes": successes,
                "image_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
