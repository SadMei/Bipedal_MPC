#!/usr/bin/env python3
"""Plot Experiment 1 prediction and tracking metrics on paired common windows."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "record" / "lambda_filter_turn_exp1_20260729_093310"
OUT_DIR = ROOT / "figures" / "manuscript_current"
SUMMARY_PATH = DATA_DIR / "paired_common_window_summary.csv"
FIGURE_PATH = OUT_DIR / "exp1_lambda_tracking_rmse.png"
VX_FIGURE_PATH = OUT_DIR / "exp1_lambda_vx_tracking_candidate.png"

START_TIME = 4.0
HORIZON_STEPS = 10
SRBM = "#2E4780"
IRCMPC = "#CC6F47"
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
AXIS = "#D7DBE7"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.65,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def rms(sum_sq: float, count: int) -> float:
    if count == 0:
        raise RuntimeError("metric window contains no samples")
    return math.sqrt(sum_sq / count)


def tracking_rmse(path: Path, terminal_time: float) -> tuple[float, float]:
    wz_sum_sq = 0.0
    vx_sum_sq = 0.0
    count = 0
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        time_i = header.index("time")
        wz_i = header.index("wz")
        wz_ref_i = header.index("wz_ref")
        vx_i = header.index("vx")
        vx_ref_i = header.index("vx_ref")
        for row in reader:
            time = float(row[time_i])
            if time < START_TIME:
                continue
            if time > terminal_time:
                break
            wz_error = float(row[wz_i]) - float(row[wz_ref_i])
            vx_error = float(row[vx_i]) - float(row[vx_ref_i])
            if math.isfinite(wz_error) and math.isfinite(vx_error):
                wz_sum_sq += wz_error * wz_error
                vx_sum_sq += vx_error * vx_error
                count += 1
    return rms(wz_sum_sq, count), rms(vx_sum_sq, count)


def prediction_rmse(path: Path, terminal_time: float) -> float:
    sum_sq = 0.0
    count = 0
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        origin_i = header.index("origin_time")
        target_i = header.index("target_time")
        step_i = header.index("horizon_steps")
        error_i = header.index("err_wz")
        for row in reader:
            if int(row[step_i]) != HORIZON_STEPS:
                continue
            origin = float(row[origin_i])
            target = float(row[target_i])
            if origin < START_TIME or target > terminal_time:
                continue
            error = float(row[error_i])
            if math.isfinite(error):
                sum_sq += error * error
                count += 1
    return rms(sum_sq, count)


def compute_summary() -> list[dict[str, object]]:
    with (DATA_DIR / "trials.csv").open(newline="") as stream:
        trials = list(csv.DictReader(stream))
    indexed = {
        (float(row["lambda_scale"]), int(row["rep"]), row["controller"]): row
        for row in trials
    }
    lambdas = sorted({key[0] for key in indexed})
    per_model: dict[tuple[float, str], list[tuple[float, float, float, float]]] = defaultdict(list)

    for lambda_scale in lambdas:
        for rep in range(1, 6):
            srbm = indexed[(lambda_scale, rep, "srbm")]
            ircmpc = indexed[(lambda_scale, rep, "vicm")]
            terminal_time = min(float(srbm["final_time"]), float(ircmpc["final_time"]))
            for model, row in (("SRBM", srbm), ("IR-CMPC", ircmpc)):
                case = row["case"]
                wz_tracking, vx_tracking = tracking_rmse(
                    DATA_DIR / f"{case}_trace.csv", terminal_time
                )
                prediction = prediction_rmse(
                    DATA_DIR / f"{case}_mpc_horizon.csv", terminal_time
                )
                per_model[(lambda_scale, model)].append(
                    (float(row["final_time"]), prediction, wz_tracking, vx_tracking)
                )

    rows: list[dict[str, object]] = []
    for lambda_scale in lambdas:
        for model in ("SRBM", "IR-CMPC"):
            values = per_model[(lambda_scale, model)]
            survival = [value[0] for value in values]
            prediction = [value[1] for value in values]
            wz_tracking = [value[2] for value in values]
            vx_tracking = [value[3] for value in values]
            rows.append(
                {
                    "lambda_scale": lambda_scale,
                    "model": model,
                    "n": len(values),
                    "mean_survival_time": statistics.mean(survival),
                    "sd_survival_time": statistics.stdev(survival),
                    "mean_h10_wz_prediction_rmse": statistics.mean(prediction),
                    "sd_h10_wz_prediction_rmse": statistics.stdev(prediction),
                    "mean_wz_tracking_rmse": statistics.mean(wz_tracking),
                    "sd_wz_tracking_rmse": statistics.stdev(wz_tracking),
                    "mean_vx_tracking_rmse": statistics.mean(vx_tracking),
                    "sd_vx_tracking_rmse": statistics.stdev(vx_tracking),
                }
            )

    with SUMMARY_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def clean_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def plot(rows: list[dict[str, object]]) -> None:
    grouped = {
        model: sorted(
            (
                row
                for row in rows
                if row["model"] == model and float(row["lambda_scale"]) <= 2.2
            ),
            key=lambda row: float(row["lambda_scale"]),
        )
        for model in ("SRBM", "IR-CMPC")
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharex=True)
    metrics = (
        (
            "mean_h10_wz_prediction_rmse",
            "sd_h10_wz_prediction_rmse",
            "10-step " + r"$\omega_z$" + " pred. RMSE [rad/s]",
            "(a)",
        ),
        (
            "mean_vx_tracking_rmse",
            "sd_vx_tracking_rmse",
            r"$v_x$ tracking RMSE [m/s]",
            "(b)",
        ),
        (
            "mean_wz_tracking_rmse",
            "sd_wz_tracking_rmse",
            r"$\omega_z$ tracking RMSE [rad/s]",
            "(c)",
        ),
    )
    styles = {
        "SRBM": (SRBM, "o", "-", "SRBM"),
        "IR-CMPC": (IRCMPC, "s", "--", "IR-CMPC (Ours)"),
    }
    for ax, (mean_key, sd_key, ylabel, panel) in zip(axes, metrics):
        for model in ("SRBM", "IR-CMPC"):
            points = grouped[model]
            x = np.asarray([float(row["lambda_scale"]) for row in points])
            mean = np.asarray([float(row[mean_key]) for row in points])
            sd = np.asarray([float(row[sd_key]) for row in points])
            color, marker, linestyle, label = styles[model]
            lower = np.maximum(0.0, mean - sd)
            upper = mean + sd
            ax.fill_between(x, lower, upper, color=color, alpha=0.13, linewidth=0)
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=3.0,
                label=label,
                zorder=3,
            )
        ax.set_ylabel(ylabel)
        ax.text(-0.02, 1.06, panel, transform=ax.transAxes, ha="left", va="bottom")
        ax.set_xlim(0.98, 2.22)
        clean_axes(ax)

    axes[0].legend(loc="upper left", frameon=False, ncol=1, handlelength=1.7)
    for ax in axes:
        ax.set_xlabel(r"Scale $\lambda$")
    for ax in axes:
        ax.set_xticks([1.0, 1.4, 1.8, 2.2])
    fig.subplots_adjust(wspace=0.34)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURE_PATH.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for model in ("SRBM", "IR-CMPC"):
        points = [
            row for row in grouped[model] if float(row["lambda_scale"]) <= 2.2
        ]
        x = np.asarray([float(row["lambda_scale"]) for row in points])
        mean = np.asarray([float(row["mean_vx_tracking_rmse"]) for row in points])
        sd = np.asarray([float(row["sd_vx_tracking_rmse"]) for row in points])
        color, marker, linestyle, label = styles[model]
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            mean,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.2,
            markersize=3.0,
            label=label,
            zorder=3,
        )
    ax.set_ylabel(r"$v_x$ tracking RMSE [m/s]")
    ax.set_xlabel(r"Leg mass/inertia scale $\lambda$")
    ax.set_xlim(0.98, 2.22)
    ax.set_xticks(np.arange(1.0, 2.21, 0.2))
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.0)
    clean_axes(ax)
    fig.savefig(VX_FIGURE_PATH, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(VX_FIGURE_PATH.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    set_style()
    rows = compute_summary()
    plot(rows)
    print(SUMMARY_PATH)
    print(FIGURE_PATH)
    print(VX_FIGURE_PATH)


if __name__ == "__main__":
    main()
