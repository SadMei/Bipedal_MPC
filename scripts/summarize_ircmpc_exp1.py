#!/usr/bin/env python3
"""Summarize the IR-CMPC branch of an Experiment 1 lambda sweep."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = (
    ("final_time", "Survival time (s)"),
    ("rms_wz_err", r"Yaw-rate tracking RMSE (rad/s)"),
    ("rms_vx_err", r"Forward-velocity tracking RMSE (m/s)"),
)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--controller", default="vicm")
    parser.add_argument("--label", default="IR-CMPC")
    parser.add_argument("--prefix", default="ircmpc_only_amp0p4")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    with (experiment_dir / "trials.csv").open(newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["controller"] == args.controller
        ]
    if not rows:
        raise SystemExit(f"No rows found for controller {args.controller!r}")

    lambdas = sorted({float(row["lambda_scale"]) for row in rows})
    grouped = {
        lambda_scale: [
            row for row in rows if float(row["lambda_scale"]) == lambda_scale
        ]
        for lambda_scale in lambdas
    }
    unexpected = {
        lambda_scale: len(group)
        for lambda_scale, group in grouped.items()
        if len(group) != 5
    }
    if unexpected:
        raise SystemExit(f"Expected five trials per lambda, got {unexpected}")

    summary_path = experiment_dir / f"{args.prefix}_summary.csv"
    fieldnames = [
        "lambda_scale",
        "n",
        "completed_30s",
        "mean_survival_time",
        "sample_sd_survival_time",
        "min_survival_time",
        "max_survival_time",
        "mean_wz_tracking_rmse",
        "sample_sd_wz_tracking_rmse",
        "mean_vx_tracking_rmse",
        "sample_sd_vx_tracking_rmse",
    ]
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for lambda_scale in lambdas:
            group = grouped[lambda_scale]
            survival = [float(row["final_time"]) for row in group]
            wz_rmse = [float(row["rms_wz_err"]) for row in group]
            vx_rmse = [float(row["rms_vx_err"]) for row in group]
            writer.writerow(
                {
                    "lambda_scale": f"{lambda_scale:.1f}",
                    "n": len(group),
                    "completed_30s": sum(value >= 30.0 for value in survival),
                    "mean_survival_time": statistics.mean(survival),
                    "sample_sd_survival_time": sample_std(survival),
                    "min_survival_time": min(survival),
                    "max_survival_time": max(survival),
                    "mean_wz_tracking_rmse": statistics.mean(wz_rmse),
                    "sample_sd_wz_tracking_rmse": sample_std(wz_rmse),
                    "mean_vx_tracking_rmse": statistics.mean(vx_rmse),
                    "sample_sd_vx_tracking_rmse": sample_std(vx_rmse),
                }
            )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.15, 8.0),
        sharex=True,
        constrained_layout=True,
    )
    blue = "#00629B"
    gray = "#666666"
    for ax, (metric, ylabel) in zip(axes, METRICS):
        groups = [
            [float(row[metric]) for row in grouped[lambda_scale]]
            for lambda_scale in lambdas
        ]
        means = [statistics.mean(group) for group in groups]
        stds = [sample_std(group) for group in groups]
        ax.errorbar(
            lambdas,
            means,
            yerr=stds,
            color=blue,
            marker="o",
            markersize=4.3,
            linewidth=1.35,
            capsize=3,
            label=f"{args.label}: mean $\\pm$ sample SD",
            zorder=3,
        )
        for lambda_scale in lambdas:
            group = grouped[lambda_scale]
            nominal = [float(row[metric]) for row in group if int(row["rep"]) == 1]
            uncertain = [
                float(row[metric]) for row in group if int(row["rep"]) != 1
            ]
            ax.scatter(
                [lambda_scale] * len(uncertain),
                uncertain,
                s=18,
                color=gray,
                alpha=0.55,
                edgecolors="none",
                label="Light-noise/push trials" if lambda_scale == lambdas[0] else None,
                zorder=2,
            )
            ax.scatter(
                [lambda_scale] * len(nominal),
                nominal,
                s=31,
                facecolors="white",
                edgecolors=blue,
                linewidths=1.1,
                label="Deterministic trial" if lambda_scale == lambdas[0] else None,
                zorder=4,
            )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].axhline(30.0, color="#888888", linestyle="--", linewidth=1.0)
    axes[0].text(2.29, 29.35, "30 s limit", ha="right", va="top", fontsize=8)
    axes[0].set_ylim(0.0, 32.0)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    axes[-1].set_xlabel(r"Leg mass/inertia scaling factor $\lambda$")
    axes[-1].set_xlim(0.96, 2.34)
    axes[-1].set_xticks(lambdas)

    png_path = experiment_dir / f"{args.prefix}_summary.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(summary_path)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
