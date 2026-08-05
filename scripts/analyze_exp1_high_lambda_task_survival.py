#!/usr/bin/env python3
"""Compute and plot conventional and speed-weighted survival metrics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
COLORS = {"srbm": "#2E4780", "vicm": "#CC6F47"}
LABELS = {"srbm": "SRBM", "vicm": "IR-CMPC (Ours)"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev_pop(values: list[float]) -> float:
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def speed_weighted_survival(trace_path: Path) -> tuple[float, float, float]:
    rows = read_rows(trace_path)
    if len(rows) < 2:
        raise RuntimeError(f"trace has fewer than two samples: {trace_path}")

    weighted_time = 0.0
    forward_distance = 0.0
    active_vx: list[float] = []
    previous_time = float(rows[0]["time"])
    previous_weight = 1.0
    previous_vx = max(0.0, float(rows[0]["vx"]))

    for row in rows[1:]:
        time = float(row["time"])
        dt = max(0.0, time - previous_time)
        vx = max(0.0, float(row["vx"]))
        vx_ref = max(0.0, float(row["vx_ref"]))
        weight = 1.0 if vx_ref <= 0.05 else min(1.0, vx / vx_ref)

        weighted_time += 0.5 * (previous_weight + weight) * dt
        forward_distance += 0.5 * (previous_vx + vx) * dt
        if vx_ref > 0.05:
            active_vx.append(float(row["vx"]))

        previous_time = time
        previous_weight = weight
        previous_vx = vx

    return weighted_time, forward_distance, mean(active_vx)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    trials = read_rows(result_dir / "trials.csv")
    enriched: list[dict[str, object]] = []
    for trial in trials:
        weighted, distance, active_vx = speed_weighted_survival(
            result_dir / trial["trace_path"]
        )
        enriched.append(
            {
                "lambda_scale": float(trial["lambda_scale"]),
                "controller": trial["controller"],
                "rep": int(trial["rep"]),
                "conventional_survival": float(trial["final_time"]),
                "speed_weighted_survival": weighted,
                "forward_distance": distance,
                "mean_active_vx": active_vx,
            }
        )
    write_csv(result_dir / "task_valid_trials.csv", enriched)

    groups: dict[tuple[float, str], list[dict[str, object]]] = defaultdict(list)
    for row in enriched:
        groups[(float(row["lambda_scale"]), str(row["controller"]))].append(row)

    summary: list[dict[str, object]] = []
    for (lambda_scale, controller), rows in sorted(groups.items()):
        conventional = [float(row["conventional_survival"]) for row in rows]
        weighted = [float(row["speed_weighted_survival"]) for row in rows]
        distances = [float(row["forward_distance"]) for row in rows]
        active_vx = [float(row["mean_active_vx"]) for row in rows]
        summary.append(
            {
                "lambda_scale": lambda_scale,
                "controller": controller,
                "n": len(rows),
                "mean_conventional_survival": mean(conventional),
                "std_conventional_survival": stdev_pop(conventional),
                "mean_speed_weighted_survival": mean(weighted),
                "std_speed_weighted_survival": stdev_pop(weighted),
                "mean_forward_distance": mean(distances),
                "std_forward_distance": stdev_pop(distances),
                "mean_active_vx": mean(active_vx),
                "std_active_vx": stdev_pop(active_vx),
            }
        )
    write_csv(result_dir / "task_valid_summary.csv", summary)

    by_key = {
        (float(row["lambda_scale"]), str(row["controller"])): row
        for row in summary
    }
    lambdas = sorted({float(row["lambda_scale"]) for row in summary})

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45))
    specifications = [
        (
            "mean_conventional_survival",
            "std_conventional_survival",
            "Conventional survival time [s]",
            "(a)",
        ),
        (
            "mean_speed_weighted_survival",
            "std_speed_weighted_survival",
            "Speed-weighted survival time [s]",
            "(b)",
        ),
    ]
    for axis, (mean_key, std_key, ylabel, panel) in zip(axes, specifications):
        for controller, marker, linestyle in [
            ("srbm", "o", "-"),
            ("vicm", "s", "--"),
        ]:
            means = [
                float(by_key[(lambda_scale, controller)][mean_key])
                for lambda_scale in lambdas
            ]
            errors = [
                float(by_key[(lambda_scale, controller)][std_key])
                for lambda_scale in lambdas
            ]
            axis.errorbar(
                lambdas,
                means,
                yerr=errors,
                color=COLORS[controller],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=4,
                elinewidth=0.8,
                capsize=2.5,
                label=LABELS[controller],
            )
        axis.set_xlabel(r"Leg inertia scale $\lambda$")
        axis.set_ylabel(ylabel)
        axis.set_xticks(lambdas)
        axis.grid(axis="y", color="#E6E8F0", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.0,
            1.04,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)

    output = result_dir / "high_lambda_survival_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(
        FIGURE_DIR / f"high_lambda_survival_comparison_{result_dir.name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
