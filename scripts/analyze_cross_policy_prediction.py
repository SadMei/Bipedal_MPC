#!/usr/bin/env python3
"""Cross-evaluate angular-dynamics predictors on noise-free closed-loop trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_five_step_prediction_error import calculate, read_rows


TRAJECTORY_SPECS = (
    ("SRBM trajectory", "*_srbm_r1_pred_error.csv"),
    (r"$I_G$-update trajectory", "*_vicm_ig_r1_pred_error.csv"),
    ("IR-CMPC trajectory", "*_vicm_ac_r1_pred_error.csv"),
    ("IR-CMPC-NF trajectory", "*_vicm_ac_nofilter_r1_pred_error.csv"),
)
PREDICTOR_SPECS = (
    ("SRBM", "SRBM"),
    (r"$I_G$ update", "VI-frozen"),
    ("IR-CMPC", "IR-frozen"),
    ("IR-CMPC-NF", "IR-NF-frozen"),
)


def find_unique(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{directory}: expected one file matching {pattern!r}, found {len(matches)}"
        )
    return matches[0]


def last_time(path: Path) -> float:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: empty prediction log")
    return float(rows[-1]["time"])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_for(
    rows: list[dict[str, object]],
    trajectories: list[tuple[str, Path]],
    predictors: tuple[tuple[str, str], ...],
    horizon: int,
    metric: str,
) -> np.ndarray:
    result = np.empty((len(trajectories), len(predictors)))
    for row_index, (trajectory, _) in enumerate(trajectories):
        for column_index, (_, predictor) in enumerate(predictors):
            match = next(
                row
                for row in rows
                if row["trajectory"] == trajectory
                and row["predictor_key"] == predictor
                and int(row["horizon_steps"]) == horizon
            )
            result[row_index, column_index] = float(match[metric])
    return result


def plot_matrices(
    path: Path,
    rows: list[dict[str, object]],
    trajectories: list[tuple[str, Path]],
    predictors: tuple[tuple[str, str], ...],
) -> None:
    panels = (
        (5, "omega_rmse", r"25-ms $\omega$ RMSE"),
        (5, "wz_rmse", r"25-ms $\omega_z$ RMSE"),
        (10, "omega_rmse", r"50-ms $\omega$ RMSE"),
        (10, "wz_rmse", r"50-ms $\omega_z$ RMSE"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.35), constrained_layout=True)
    trajectory_labels = [
        name.replace(" trajectory", "") for name, _ in trajectories
    ]
    predictor_labels = [name for name, _ in predictors]

    for panel_index, (axis, (horizon, metric, title)) in enumerate(
        zip(axes.flat, panels)
    ):
        values = matrix_for(rows, trajectories, predictors, horizon, metric)
        image = axis.imshow(values, cmap="Blues", aspect="auto")
        axis.set_title(title, fontsize=9)
        axis.set_xticks(
            range(len(predictors)), predictor_labels, fontsize=8, rotation=15
        )
        axis.set_yticks(range(len(trajectories)), trajectory_labels, fontsize=8)
        axis.set_xlabel("Predictor", fontsize=8)
        if panel_index % 2 == 0:
            axis.set_ylabel("Trajectory generator", fontsize=8)
        threshold = 0.65 * np.max(values)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > threshold else "black",
                )
        colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("rad/s", fontsize=8)

    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--start-time", type=float, default=4.0)
    parser.add_argument("--end-margin", type=float, default=0.25)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = []
    for trajectory, pattern in TRAJECTORY_SPECS:
        matches = sorted(args.experiment_dir.glob(pattern))
        if len(matches) == 1:
            sources.append((trajectory, matches[0]))
        elif len(matches) > 1:
            raise ValueError(
                f"{args.experiment_dir}: expected at most one file matching "
                f"{pattern!r}, found {len(matches)}"
            )
    if len(sources) < 3:
        raise ValueError("SRBM, I_G-update, and IR-CMPC trajectories are required")
    include_no_filter = any(name == "IR-CMPC-NF trajectory" for name, _ in sources)
    predictors = (
        PREDICTOR_SPECS if include_no_filter else PREDICTOR_SPECS[:3]
    )
    common_end = min(last_time(path) for _, path in sources) - args.end_margin
    if common_end <= args.start_time:
        raise ValueError("common evaluation interval is empty")

    output_dir = args.output_dir or args.experiment_dir / "cross_policy_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    pooled_trials: list[list[dict[str, str]]] = []

    for trajectory, path in sources:
        trials = read_rows([path], args.start_time, common_end)
        pooled_trials.extend(trials)
        summary, _ = calculate(trials, args.max_horizon)
        for row in summary:
            if row["model"] not in {key for _, key in predictors}:
                continue
            output_rows.append(
                {
                    "trajectory": trajectory,
                    "source_csv": path.name,
                    "evaluation_start_s": args.start_time,
                    "evaluation_end_s": common_end,
                    "horizon_steps": row["horizon_steps"],
                    "horizon_ms": row["horizon_ms"],
                    "predictor": next(
                        label for label, key in predictors if key == row["model"]
                    ),
                    "predictor_key": row["model"],
                    "samples": row["samples"],
                    "omega_rmse": row["omega_rmse"],
                    "wz_rmse": row["wz_rmse"],
                    "omega_error_p95": row["omega_error_p95"],
                }
            )

    pooled_summary, _ = calculate(pooled_trials, args.max_horizon)
    pooled_rows: list[dict[str, object]] = []
    for row in pooled_summary:
        if row["model"] not in {key for _, key in predictors}:
            continue
        pooled_rows.append(
            {
                "trajectory": "Equal pooled trajectories",
                "evaluation_start_s": args.start_time,
                "evaluation_end_s": common_end,
                "horizon_steps": row["horizon_steps"],
                "horizon_ms": row["horizon_ms"],
                "predictor": next(
                    label for label, key in predictors if key == row["model"]
                ),
                "predictor_key": row["model"],
                "samples": row["samples"],
                "omega_rmse": row["omega_rmse"],
                "wz_rmse": row["wz_rmse"],
                "omega_error_p95": row["omega_error_p95"],
            }
        )

    matrix_csv = output_dir / "cross_policy_prediction_matrix.csv"
    pooled_csv = output_dir / "cross_policy_prediction_pooled.csv"
    figure_path = output_dir / "cross_policy_prediction_matrix.png"
    write_csv(matrix_csv, output_rows)
    write_csv(pooled_csv, pooled_rows)
    plot_matrices(figure_path, output_rows, sources, predictors)

    print(f"Common interval: {args.start_time:.3f}-{common_end:.3f} s")
    horizon = args.max_horizon
    print(f"H={horizon} ({5 * horizon} ms), equal pooled trajectories")
    for row in pooled_rows:
        if int(row["horizon_steps"]) == horizon:
            print(
                f"  {row['predictor']:<12} "
                f"omega={float(row['omega_rmse']):.6f} "
                f"wz={float(row['wz_rmse']):.6f}"
            )
    print(matrix_csv)
    print(pooled_csv)
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
