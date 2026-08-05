#!/usr/bin/env python3
"""Plot the candidate four-model internal-momentum-rate ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "experiment_data" / "experiment_2_hrel" / "summary.csv"
OUTPUT = ROOT / "figures" / "exp2_hrel_ablation_candidate.png"

ORDER = ["SRBM", "IRM-CMPC", "VI-CMPC", "IR-CMPC"]
COLORS = ["#34558B", "#C43C4E", "#69B532", "#D07145"]
DISPLAY_ORDER = ["SRBM", "IRM-CMPC (Ours)", "VI-CMPC", "IR-CMPC"]


def main() -> None:
    with DATA.open(newline="") as stream:
        rows = {row["controller_label"]: row for row in csv.DictReader(stream)}
    selected = [rows[label] for label in ORDER]
    y = np.arange(len(ORDER))

    metrics = [
        ("mean_survival_time", "Survival time [s]", (0.0, 33.0), "higher"),
        (
            "mean_h10_wz_prediction_rmse",
            r"10-step $\omega_z$ prediction RMSE [rad/s]",
            (0.0, 1.3),
            "lower",
        ),
        (
            "mean_wz_tracking_rmse",
            r"$\omega_z$ tracking RMSE [rad/s]",
            (0.0, 0.58),
            "lower",
        ),
        ("mean_mpc_wall_ms", "Mean MPC update time [ms]", (0.0, 2.05), None),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.65,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.5))
    for panel, (ax, (key, xlabel, xlim, direction)) in enumerate(
        zip(axes.flat, metrics)
    ):
        values = np.asarray([float(row[key]) for row in selected])
        ax.barh(y, values, height=0.66, color=COLORS, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y, DISPLAY_ORDER)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#DDE2EA", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9AA3B2")
        ax.text(-0.13, 1.06, f"({chr(ord('a') + panel)})", transform=ax.transAxes)

        if key == "mean_survival_time":
            full = values[ORDER.index("IRM-CMPC")]
            ax.annotate(
                "reaches 30 s limit",
                xy=(full, ORDER.index("IRM-CMPC")),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=COLORS[ORDER.index("IRM-CMPC")],
                fontweight="semibold",
            )
        elif direction is not None:
            baseline = values[ORDER.index("SRBM")]
            full = values[ORDER.index("IRM-CMPC")]
            change = 100.0 * abs(full - baseline) / baseline
            ax.annotate(
                f"{change:.1f}% {direction}",
                xy=(full, ORDER.index("IRM-CMPC")),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=COLORS[ORDER.index("IRM-CMPC")],
                fontweight="semibold",
            )

    axes[0, 0].axvline(30.0, color="#7A8495", linestyle=":", linewidth=0.8)
    axes[1, 1].axvline(2.0, color="#7A8495", linestyle=":", linewidth=0.8)
    axes[1, 1].text(
        1.97,
        -0.35,
        "all below 2 ms",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#657084",
    )
    fig.tight_layout(w_pad=2.0, h_pad=1.3)
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
