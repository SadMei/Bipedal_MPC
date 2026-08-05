#!/usr/bin/env python3
"""Plot the final matched SRBM/IRM-CMPC Experiment 1 metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "record" / "exp1_two_model_final_tau001_reset_20260805"
DEFAULT_OUTPUT = ROOT / "figures" / "manuscript_current" / "exp1_lambda_tracking_rmse.png"
COLORS = {"SRBM": "#2E4780", "IR-CMPC": "#D8842F", "IRM-CMPC": "#C43C4E"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with (args.data_dir / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {
        label: sorted(
            (row for row in rows if row["controller_label"] == label),
            key=lambda row: float(row["lambda_scale"]),
        )
        for label in ("SRBM", "IRM-CMPC")
    }

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    metrics = (
        (
            "mean_h10_wz_prediction_rmse",
            "sample_sd_h10_wz_prediction_rmse",
            r"10-step $\omega_z$ prediction RMSE [rad/s]",
        ),
        (
            "mean_vx_tracking_rmse",
            "sample_sd_vx_tracking_rmse",
            r"$v_x$ tracking RMSE [m/s]",
        ),
        (
            "mean_wz_tracking_rmse",
            "sample_sd_wz_tracking_rmse",
            r"$\omega_z$ tracking RMSE [rad/s]",
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), sharex=True)
    for panel, (ax, (mean_key, sd_key, ylabel)) in enumerate(zip(axes, metrics)):
        for label, marker, linestyle in (
            ("SRBM", "o", "-"),
            ("IRM-CMPC", "s", "-."),
        ):
            points = grouped[label]
            x = np.asarray([float(row["lambda_scale"]) for row in points])
            mean = np.asarray([float(row[mean_key]) for row in points])
            sd = np.asarray([float(row[sd_key]) for row in points])
            color = COLORS[label]
            ax.fill_between(
                x,
                np.maximum(0.0, mean - sd),
                mean + sd,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.2,
                markersize=3.0,
                label="IRM-CMPC (Ours)" if label == "IRM-CMPC" else label,
                zorder=3,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Scale $\lambda$")
        ax.set_xlim(0.98, 2.22)
        ax.set_xticks([1.0, 1.4, 1.8, 2.2])
        ax.grid(axis="y", color="#E6E8F0", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.02, 1.06, f"({chr(ord('a') + panel)})", transform=ax.transAxes)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.7)
    fig.subplots_adjust(wspace=0.34)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
