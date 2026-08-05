#!/usr/bin/env python3
"""Plot the complete matched Experiment 1 sweep for three MPC models."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = (
    ("srbm", "SRBM", "#4D4D4D", "s", -0.027),
    ("ir_cmpc", r"IR-CMPC (frozen $I_G$)", "#0072B2", "o", -0.009),
    ("ir_linear", r"IR-CMPC (linear $I_G$)", "#009E73", "D", 0.009),
    ("dm_cmpc", "DM-CMPC", "#D55E00", "^", 0.027),
)
METRICS = (
    ("final_time", "Survival time (s)"),
    ("rms_wz_err", r"Yaw-rate tracking RMSE (rad/s)"),
    ("rms_vx_err", r"Forward-velocity tracking RMSE (m/s)"),
    (
        "h10_wz_prediction_rmse",
        r"10-step yaw-rate prediction RMSE (rad/s)",
    ),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def h10_wz_prediction_rmse(path: Path, start_time: float = 4.0) -> float:
    errors: list[float] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                if int(row["horizon_steps"]) != 10:
                    continue
                if float(row["origin_time"]) < start_time:
                    continue
                error = float(row["err_wz"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(error):
                errors.append(error)
    if not errors:
        raise RuntimeError(f"No valid 10-step yaw-rate errors in {path}")
    return math.sqrt(statistics.mean(error * error for error in errors))


def add_h10_metrics(rows: list[dict[str, str]], directory: Path) -> None:
    for row in rows:
        horizon_path = directory / f"{row['case']}_mpc_horizon.csv"
        row["h10_wz_prediction_rmse"] = str(
            h10_wz_prediction_rmse(horizon_path)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("srbm_ir_dir", type=Path)
    parser.add_argument("dm_dir", type=Path)
    parser.add_argument("linear_ir_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    srbm_ir_dir = args.srbm_ir_dir.resolve()
    dm_dir = args.dm_dir.resolve()
    linear_ir_dir = args.linear_ir_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else dm_dir / "full_model_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_rows(srbm_ir_dir / "trials.csv")
    dm_rows = read_rows(dm_dir / "trials.csv")
    linear_ir_rows = read_rows(linear_ir_dir / "trials.csv")
    add_h10_metrics(base_rows, srbm_ir_dir)
    add_h10_metrics(dm_rows, dm_dir)
    add_h10_metrics(linear_ir_rows, linear_ir_dir)
    rows_by_model = {
        "srbm": [row for row in base_rows if row["controller"] == "srbm"],
        "ir_cmpc": [row for row in base_rows if row["controller"] == "vicm"],
        "ir_linear": [
            row for row in linear_ir_rows if row["controller"] == "vicm"
        ],
        "dm_cmpc": [row for row in dm_rows if row["controller"] == "vicm"],
    }
    lambdas = sorted(
        {float(row["lambda_scale"]) for row in rows_by_model["srbm"]}
    )
    for model, rows in rows_by_model.items():
        model_lambdas = sorted({float(row["lambda_scale"]) for row in rows})
        if model_lambdas != lambdas:
            raise RuntimeError(f"Lambda grid mismatch for {model}: {model_lambdas}")
        counts = {
            lambda_scale: sum(
                abs(float(row["lambda_scale"]) - lambda_scale) < 1.0e-9
                for row in rows
            )
            for lambda_scale in lambdas
        }
        if set(counts.values()) != {5}:
            raise RuntimeError(f"Expected five trials per lambda for {model}: {counts}")

    summary_path = output_dir / "full_model_summary.csv"
    summary_fields = [
        "lambda_scale",
        "model",
        "n",
        "completed_30s",
        "mean_survival_time",
        "sample_sd_survival_time",
        "mean_wz_tracking_rmse",
        "sample_sd_wz_tracking_rmse",
        "mean_vx_tracking_rmse",
        "sample_sd_vx_tracking_rmse",
        "mean_h10_wz_prediction_rmse",
        "sample_sd_h10_wz_prediction_rmse",
    ]
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for model, label, _, _, _ in MODELS:
            for lambda_scale in lambdas:
                group = [
                    row
                    for row in rows_by_model[model]
                    if abs(float(row["lambda_scale"]) - lambda_scale) < 1.0e-9
                ]
                survival = [float(row["final_time"]) for row in group]
                wz_rmse = [float(row["rms_wz_err"]) for row in group]
                vx_rmse = [float(row["rms_vx_err"]) for row in group]
                h10_wz_rmse = [
                    float(row["h10_wz_prediction_rmse"]) for row in group
                ]
                writer.writerow(
                    {
                        "lambda_scale": f"{lambda_scale:.1f}",
                        "model": label,
                        "n": len(group),
                        "completed_30s": sum(value >= 30.0 for value in survival),
                        "mean_survival_time": statistics.mean(survival),
                        "sample_sd_survival_time": sample_std(survival),
                        "mean_wz_tracking_rmse": statistics.mean(wz_rmse),
                        "sample_sd_wz_tracking_rmse": sample_std(wz_rmse),
                        "mean_vx_tracking_rmse": statistics.mean(vx_rmse),
                        "sample_sd_vx_tracking_rmse": sample_std(vx_rmse),
                        "mean_h10_wz_prediction_rmse": statistics.mean(
                            h10_wz_rmse
                        ),
                        "sample_sd_h10_wz_prediction_rmse": sample_std(
                            h10_wz_rmse
                        ),
                    }
                )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(7.2, 10.5),
        sharex=True,
        constrained_layout=True,
    )
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for model, label, color, marker, offset in MODELS:
            groups = []
            for lambda_scale in lambdas:
                groups.append(
                    [
                        float(row[metric])
                        for row in rows_by_model[model]
                        if abs(float(row["lambda_scale"]) - lambda_scale) < 1.0e-9
                    ]
                )
            means = [statistics.mean(group) for group in groups]
            stds = [sample_std(group) for group in groups]
            x = [lambda_scale + offset for lambda_scale in lambdas]
            ax.errorbar(
                x,
                means,
                yerr=stds,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.35,
                capsize=2.5,
                label=label,
                zorder=3,
            )
            for x_value, group in zip(x, groups):
                ax.scatter(
                    [x_value] * len(group),
                    group,
                    s=13,
                    color=color,
                    alpha=0.25,
                    edgecolors="none",
                    zorder=2,
                )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].axhline(30.0, color="#888888", linestyle="--", linewidth=1.0)
    axes[0].text(2.29, 29.35, "30 s limit", ha="right", va="top", fontsize=8)
    axes[0].set_ylim(0.0, 32.0)
    axes[0].legend(frameon=False, ncol=2, fontsize=7.5, loc="lower left")
    axes[-1].set_xlabel(r"Leg mass/inertia scaling factor $\lambda$")
    axes[-1].set_xlim(0.95, 2.35)
    axes[-1].set_xticks(lambdas)

    png_path = output_dir / "full_model_comparison.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    survival_fig, survival_ax = plt.subplots(
        figsize=(7.2, 3.9), constrained_layout=True
    )
    for model, label, color, marker, offset in MODELS:
        groups = []
        for lambda_scale in lambdas:
            groups.append(
                [
                    float(row["final_time"])
                    for row in rows_by_model[model]
                    if abs(float(row["lambda_scale"]) - lambda_scale) < 1.0e-9
                ]
            )
        means = [statistics.mean(group) for group in groups]
        stds = [sample_std(group) for group in groups]
        x = [lambda_scale + offset for lambda_scale in lambdas]
        survival_ax.errorbar(
            x,
            means,
            yerr=stds,
            color=color,
            marker=marker,
            markersize=4.8,
            linewidth=1.4,
            capsize=2.5,
            label=label,
            zorder=3,
        )
        for x_value, group in zip(x, groups):
            survival_ax.scatter(
                [x_value] * len(group),
                group,
                s=15,
                color=color,
                alpha=0.25,
                edgecolors="none",
                zorder=2,
            )
    survival_ax.axhline(
        30.0, color="#888888", linestyle="--", linewidth=1.0
    )
    survival_ax.text(
        2.29, 29.35, "30 s limit", ha="right", va="top", fontsize=8
    )
    survival_ax.set_xlabel(r"Leg mass/inertia scaling factor $\lambda$")
    survival_ax.set_ylabel("Survival time (s)")
    survival_ax.set_xlim(0.95, 2.35)
    survival_ax.set_ylim(0.0, 32.0)
    survival_ax.set_xticks(lambdas)
    survival_ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    survival_ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="lower left")
    for spine in ("top", "right"):
        survival_ax.spines[spine].set_visible(False)
    survival_png_path = output_dir / "full_model_survival.png"
    survival_pdf_path = survival_png_path.with_suffix(".pdf")
    survival_fig.savefig(survival_png_path, dpi=300)
    survival_fig.savefig(survival_pdf_path)

    print(summary_path)
    print(png_path)
    print(pdf_path)
    print(survival_png_path)
    print(survival_pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
