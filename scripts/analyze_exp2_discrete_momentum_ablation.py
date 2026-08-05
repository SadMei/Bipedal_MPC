#!/usr/bin/env python3
"""Analyze Experiment 2 on controller-specific and common time windows."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


CONTROLLERS = ("srbm", "vicm_ig", "dm_frozen", "dm_preview")
LABELS = {
    "srbm": "SRBM",
    "vicm_ig": "VI-CMPC",
    "dm_frozen": "DM-CMPC-FI",
    "dm_preview": "DM-CMPC",
}
COLORS = ("#5B5B5B", "#4E79A7", "#E07B39", "#0072B2")


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--eval-start", type=float, default=4.0)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    with (experiment_dir / "trials.csv").open(newline="") as stream:
        trials = list(csv.DictReader(stream))

    common_rows: list[dict[str, object]] = []
    repetitions = sorted({int(row["rep"]) for row in trials})
    for rep in repetitions:
        group = [row for row in trials if int(row["rep"]) == rep]
        common_end = min(float(row["final_time"]) for row in group)
        for controller in CONTROLLERS:
            trial = next(row for row in group if row["controller"] == controller)
            wz_error: list[float] = []
            tracking_error: list[float] = []
            with (experiment_dir / trial["trace_path"]).open(newline="") as stream:
                for trace_row in csv.DictReader(stream):
                    time_value = float(trace_row["time"])
                    if args.eval_start <= time_value <= common_end:
                        wz_error.append(
                            float(trace_row["wz"]) - float(trace_row["wz_ref"])
                        )
                        tracking_error.append(float(trace_row["vel_track_error"]))
            common_rows.append(
                {
                    "rep": rep,
                    "controller": controller,
                    "controller_label": LABELS[controller],
                    "common_end": common_end,
                    "rms_wz_err": rms(wz_error),
                    "rms_tracking_err": rms(tracking_error),
                }
            )

    with (experiment_dir / "common_window_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(common_rows[0]))
        writer.writeheader()
        writer.writerows(common_rows)

    summary: list[dict[str, object]] = []
    for controller in CONTROLLERS:
        original = [row for row in trials if row["controller"] == controller]
        common = [row for row in common_rows if row["controller"] == controller]
        survival = [float(row["final_time"]) for row in original]
        wz = [float(row["rms_wz_err"]) for row in common]
        tracking = [float(row["rms_tracking_err"]) for row in common]
        summary.append(
            {
                "controller": controller,
                "controller_label": LABELS[controller],
                "mean_survival_time": statistics.mean(survival),
                "std_survival_time": statistics.pstdev(survival),
                "mean_common_rms_wz_err": statistics.mean(wz),
                "std_common_rms_wz_err": statistics.pstdev(wz),
                "mean_common_rms_tracking_err": statistics.mean(tracking),
                "std_common_rms_tracking_err": statistics.pstdev(tracking),
            }
        )

    with (experiment_dir / "common_window_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.35), constrained_layout=True)
    specs = (
        ("mean_survival_time", "std_survival_time", "Survival time (s)"),
        (
            "mean_common_rms_wz_err",
            "std_common_rms_wz_err",
            r"Common-window $w_z$ RMS (rad/s)",
        ),
        (
            "mean_common_rms_tracking_err",
            "std_common_rms_tracking_err",
            "Common-window velocity RMS (m/s)",
        ),
    )
    x = list(range(len(CONTROLLERS)))
    for ax, (mean_field, std_field, ylabel) in zip(axes, specs):
        means = [float(row[mean_field]) for row in summary]
        stds = [float(row[std_field]) for row in summary]
        ax.bar(x, means, yerr=stds, color=COLORS, capsize=3, width=0.68)
        ax.set_xticks(x, [LABELS[controller] for controller in CONTROLLERS])
        ax.tick_params(axis="x", rotation=22)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    output = experiment_dir / "exp2_discrete_momentum_ablation.png"
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
