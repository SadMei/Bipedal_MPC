#!/usr/bin/env python3
"""Plot a matched deterministic comparison of SRBM, IR-CMPC, and discrete MPC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LAMBDA_VALUES = (1.6, 1.7, 1.8, 2.1)
STYLES = {
    "srbm": ("SRBM", "#4D4D4D", "s"),
    "ir_cmpc": ("IR-CMPC", "#0072B2", "o"),
    "discrete": ("Discrete angular-momentum MPC", "#D55E00", "^"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def find_row(
    rows: list[dict[str, str]],
    controller: str,
    lambda_scale: float,
    rep: int = 1,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["controller"] == controller
        and int(row["rep"]) == rep
        and abs(float(row["lambda_scale"]) - lambda_scale) < 1.0e-9
        and int(row["noise_enabled"]) == 0
        and abs(float(row["push_force"])) < 1.0e-9
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one deterministic {controller} row at "
            f"lambda={lambda_scale}, found {len(matches)}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("continuous_dir", type=Path)
    parser.add_argument("discrete_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    continuous_dir = args.continuous_dir.resolve()
    discrete_dir = args.discrete_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else continuous_dir / "matched_three_model_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    continuous_rows = read_rows(continuous_dir / "trials.csv")
    discrete_rows = read_rows(discrete_dir / "trials.csv")
    data: dict[str, list[dict[str, str]]] = {
        "ir_cmpc": [
            find_row(continuous_rows, "vicm", lambda_scale)
            for lambda_scale in LAMBDA_VALUES
        ],
        "srbm": [
            find_row(discrete_rows, "srbm", lambda_scale)
            for lambda_scale in LAMBDA_VALUES
        ],
        "discrete": [
            find_row(discrete_rows, "vicm", lambda_scale)
            for lambda_scale in LAMBDA_VALUES
        ],
    }

    csv_path = output_dir / "matched_deterministic_comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["lambda_scale", "model", "survival_time", "wz_tracking_rmse"]
        )
        for lambda_scale_index, lambda_scale in enumerate(LAMBDA_VALUES):
            for model in ("srbm", "ir_cmpc", "discrete"):
                row = data[model][lambda_scale_index]
                writer.writerow(
                    [
                        lambda_scale,
                        STYLES[model][0],
                        row["final_time"],
                        row["rms_wz_err"],
                    ]
                )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.15, 5.9),
        sharex=True,
        constrained_layout=True,
    )
    for model in ("srbm", "ir_cmpc", "discrete"):
        label, color, marker = STYLES[model]
        axes[0].plot(
            LAMBDA_VALUES,
            [float(row["final_time"]) for row in data[model]],
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(
            LAMBDA_VALUES,
            [float(row["rms_wz_err"]) for row in data[model]],
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.5,
            label=label,
        )

    axes[0].axhline(30.0, color="#888888", linestyle="--", linewidth=1.0)
    axes[0].text(2.09, 29.35, "30 s limit", ha="right", va="top", fontsize=8)
    axes[0].set_ylabel("Survival time (s)")
    axes[0].set_ylim(0.0, 32.0)
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    axes[1].set_ylabel(r"Yaw-rate tracking RMSE (rad/s)")
    axes[1].set_xlabel(r"Leg mass/inertia scaling factor $\lambda$")
    axes[1].set_xticks(LAMBDA_VALUES)
    axes[1].set_xlim(1.57, 2.13)
    for ax in axes:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    png_path = output_dir / "matched_deterministic_three_model.png"
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(csv_path)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
