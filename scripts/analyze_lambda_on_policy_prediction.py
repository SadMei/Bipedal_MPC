#!/usr/bin/env python3
"""Analyze each controller's own MPC horizon predictions across lambda."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from analyze_on_policy_mpc_prediction import read_rows, summarize


LABELS = {"srbm": "SRBM", "vicm": "Full inertia-rate model"}
COLORS = {"srbm": "#4D4D4D", "vicm": "#1769AA"}
MARKERS = {"srbm": "o", "vicm": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_path(record_dir: Path, case: str) -> Path:
    return record_dir / f"mpc_horizon_exp1_{case}_lf0.500000.csv"


def plot(path: Path, rows: list[dict[str, object]], horizon: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    selected_horizon = [
        row for row in rows if int(row["horizon_steps"]) == horizon
    ]
    for controller in ("srbm", "vicm"):
        selected = sorted(
            (row for row in selected_horizon if row["controller"] == controller),
            key=lambda row: float(row["lambda_scale"]),
        )
        lambdas = [float(row["lambda_scale"]) for row in selected]
        axes[0].plot(
            lambdas,
            [float(row["omega_rmse"]) for row in selected],
            color=COLORS[controller],
            marker=MARKERS[controller],
            linewidth=1.3,
            markersize=4,
            label=LABELS[controller],
        )
        axes[1].plot(
            lambdas,
            [float(row["wz_rmse"]) for row in selected],
            color=COLORS[controller],
            marker=MARKERS[controller],
            linewidth=1.3,
            markersize=4,
            label=LABELS[controller],
        )
    axes[0].set_ylabel(r"$\omega$ RMSE (rad s$^{-1}$)")
    axes[1].set_ylabel(r"$\omega_z$ RMSE (rad s$^{-1}$)")
    for axis in axes:
        axis.set_xlabel(r"Leg inertia scale $\lambda$")
        axis.grid(True, color="#D9D9D9", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_title(f"{5 * horizon}-ms on-policy prediction")
    axes[1].set_title(f"{5 * horizon}-ms on-policy yaw-rate prediction")
    axes[0].legend(frameon=False)
    figure.tight_layout(pad=0.7, w_pad=1.2)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--start-time", type=float, default=8.0)
    parser.add_argument("--end-margin", type=float, default=0.25)
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Minimum common duration, normally one complete turn period.",
    )
    parser.add_argument("--max-horizon", type=int, default=10)
    args = parser.parse_args()

    record_dir = args.experiment_dir.parent
    trials = read_csv(args.experiment_dir / "trials.csv")
    grouped: dict[float, dict[str, Path]] = {}
    for row in trials:
        controller = row["controller"]
        if controller not in LABELS or int(row["rep"]) != 1:
            continue
        path = source_path(record_dir, row["case"])
        if not path.is_file():
            raise FileNotFoundError(path)
        grouped.setdefault(float(row["lambda_scale"]), {})[controller] = path

    output_rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for lambda_scale, paths in sorted(grouped.items()):
        if set(paths) != set(LABELS):
            excluded.append(
                {
                    "lambda_scale": lambda_scale,
                    "reason": "missing paired controller horizon log",
                }
            )
            continue
        controller_rows = {
            controller: read_rows(path) for controller, path in paths.items()
        }
        common_end = min(
            max(row["actual_time"] for row in rows)
            for rows in controller_rows.values()
        ) - args.end_margin
        duration = common_end - args.start_time
        if duration < args.min_duration:
            excluded.append(
                {
                    "lambda_scale": lambda_scale,
                    "reason": (
                        f"common post-transient duration {duration:.3f} s "
                        f"< {args.min_duration:.3f} s"
                    ),
                }
            )
            continue
        for controller, rows in controller_rows.items():
            metrics = summarize(rows, args.start_time, common_end)
            for metric in metrics:
                if int(metric["horizon_steps"]) > args.max_horizon:
                    continue
                output_rows.append(
                    {
                        "lambda_scale": lambda_scale,
                        "controller": controller,
                        "controller_label": LABELS[controller],
                        "evaluation_start_s": args.start_time,
                        "evaluation_end_s": common_end,
                        "evaluation_duration_s": duration,
                        **metric,
                    }
                )

    if not output_rows:
        raise RuntimeError("no lambda point has a full post-transient period")
    summary_path = args.experiment_dir / "lambda_on_policy_prediction.csv"
    excluded_path = args.experiment_dir / "lambda_on_policy_excluded.csv"
    figure_path = args.experiment_dir / "lambda_on_policy_prediction.png"
    write_csv(summary_path, output_rows)
    write_csv(excluded_path, excluded)
    plot(figure_path, output_rows, args.max_horizon)

    print("lambda  end_s  SRBM_H10  Full_H10  reduction  SRBM_wz  Full_wz")
    lambdas = sorted({float(row["lambda_scale"]) for row in output_rows})
    for lambda_scale in lambdas:
        selected = {
            str(row["controller"]): row
            for row in output_rows
            if float(row["lambda_scale"]) == lambda_scale
            and int(row["horizon_steps"]) == args.max_horizon
        }
        srbm = selected["srbm"]
        full = selected["vicm"]
        srbm_error = float(srbm["omega_rmse"])
        full_error = float(full["omega_rmse"])
        print(
            f"{lambda_scale:4.1f}  {float(srbm['evaluation_end_s']):5.2f}  "
            f"{srbm_error:9.5f}  {full_error:9.5f}  "
            f"{100.0 * (srbm_error - full_error) / srbm_error:8.2f}%  "
            f"{float(srbm['wz_rmse']):8.5f}  {float(full['wz_rmse']):8.5f}"
        )
    print(summary_path)
    print(figure_path)
    print(f"Excluded lambda points: {len(excluded)}; see {excluded_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
