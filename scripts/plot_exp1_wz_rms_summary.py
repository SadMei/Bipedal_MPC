#!/usr/bin/env python3
"""Plot Experiment 1 yaw-rate tracking RMS with all repeated trials."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"srbm": "#4D4D4D", "vicm": "#0072B2"}
LABELS = {"srbm": "SRBM", "vicm": "IR-CMPC"}
OFFSETS = {"srbm": -0.012, "vicm": 0.012}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    output = args.output or experiment_dir / "wz_tracking_rms_overview.png"
    with (experiment_dir / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=True)
    for controller in ("srbm", "vicm"):
        lambdas = sorted(
            {float(row["lambda_scale"]) for row in rows if row["controller"] == controller}
        )
        groups = [
            [
                float(row["rms_wz_err"])
                for row in rows
                if row["controller"] == controller
                and float(row["lambda_scale"]) == lambda_scale
            ]
            for lambda_scale in lambdas
        ]
        means = [statistics.mean(group) for group in groups]
        stds = [statistics.pstdev(group) for group in groups]
        x = [value + OFFSETS[controller] for value in lambdas]
        ax.errorbar(
            x,
            means,
            yerr=stds,
            color=COLORS[controller],
            marker="o",
            markersize=4.5,
            linewidth=1.4,
            capsize=3,
            label=LABELS[controller],
            zorder=3,
        )
        for x_value, group in zip(x, groups):
            ax.scatter(
                [x_value] * len(group),
                group,
                s=14,
                color=COLORS[controller],
                alpha=0.35,
                edgecolors="none",
                zorder=2,
            )

    ax.set_xlabel(r"Leg mass/inertia scaling factor $\lambda$")
    ax.set_ylabel(r"RMS yaw-rate tracking error (rad/s)")
    ax.set_xlim(0.96, 2.34)
    ax.set_ylim(bottom=0.35)
    ax.set_xticks([1.0 + 0.1 * index for index in range(14)])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
