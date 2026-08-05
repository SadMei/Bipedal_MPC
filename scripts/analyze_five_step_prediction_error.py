#!/usr/bin/env python3
"""Evaluate angular-velocity rollouts at the 10-step MPC horizon."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("SRBM", "VI-frozen", "IR-frozen", "IR-linear", "IR-NF-frozen")
COLORS = {
    "SRBM": "#5B6573",
    "VI-frozen": "#68A7C4",
    "IR-frozen": "#1769AA",
    "IR-linear": "#174A7E",
    "IR-NF-frozen": "#C45A3C",
}


def read_rows(
    paths: list[Path], start_time: float, end_time: float
) -> list[list[dict[str, str]]]:
    trials: list[list[dict[str, str]]] = []
    required = {
        "start_wx",
        "nominal_i00",
        "inertia_i00",
        "idot_filtered_i00",
        "idot_raw_i00",
    }
    for path in paths:
        with path.open(newline="") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if start_time <= float(row["time"]) <= end_time
            ]
        if not rows:
            raise ValueError(f"{path}: no rows at or after {start_time:g} s")
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(
                f"{path}: missing rollout fields: {', '.join(sorted(missing))}"
            )
        trials.append(rows)
    return trials


def vector(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [float(row[f"{prefix}_wx"]), float(row[f"{prefix}_wy"]), float(row[f"{prefix}_wz"])],
        dtype=float,
    )


def matrix(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [[float(row[f"{prefix}_i{r}{c}"]) for c in range(3)] for r in range(3)],
        dtype=float,
    )


def rz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rollout(window: list[dict[str, str]], model: str) -> np.ndarray:
    first = window[0]
    omega = vector(first, "start")
    rpy = np.array(
        [
            float(first["start_roll"]),
            float(first["start_pitch"]),
            float(first["start_yaw"]),
        ],
        dtype=float,
    )
    nominal = matrix(first, "nominal")
    inertia = matrix(first, "inertia")
    idot_filtered = matrix(first, "idot_filtered")
    idot_raw = matrix(first, "idot_raw")
    elapsed_time = 0.0

    for row in window:
        dt = float(row["dt"])
        angular_impulse = np.array(
            [
                float(row["moment_impulse_x"]),
                float(row["moment_impulse_y"]),
                float(row["moment_impulse_z"]),
            ],
            dtype=float,
        )
        omega_before = omega.copy()
        if model == "SRBM":
            rotation = rz(rpy[2])
            inertia_step = rotation @ nominal @ rotation.T
            idot_step = np.zeros((3, 3))
        elif model == "VI-frozen":
            inertia_step = inertia
            idot_step = np.zeros((3, 3))
        elif model == "IR-frozen":
            inertia_step = inertia
            idot_step = idot_filtered
        elif model == "IR-linear":
            inertia_step = inertia + elapsed_time * idot_filtered
            inertia_step = 0.5 * (inertia_step + inertia_step.T)
            idot_step = idot_filtered
        elif model == "IR-NF-frozen":
            inertia_step = inertia
            idot_step = idot_raw
        else:
            raise ValueError(model)

        omega = omega + np.linalg.solve(
            inertia_step,
            angular_impulse - idot_step @ omega * dt,
        )
        rpy = rpy + dt * (rz(rpy[2]).T @ omega_before)
        elapsed_time += dt
    return omega


def contiguous(window: list[dict[str, str]], tolerance: float = 5.0e-4) -> bool:
    if len(window) < 2:
        return True
    for previous, current in zip(window, window[1:]):
        expected = float(previous["time"]) + float(current["dt"])
        if abs(float(current["time"]) - expected) > tolerance:
            return False
    return True


def calculate(
    trials: list[list[dict[str, str]]], max_horizon: int
) -> tuple[list[dict[str, object]], dict[tuple[int, str], list[np.ndarray]]]:
    errors: dict[tuple[int, str], list[np.ndarray]] = {
        (horizon, model): []
        for horizon in range(1, max_horizon + 1)
        for model in MODELS
    }
    for rows in trials:
        for start in range(len(rows)):
            first = rows[start]
            start_omega = vector(first, "start")
            start_rpy = np.array(
                [
                    float(first["start_roll"]),
                    float(first["start_pitch"]),
                    float(first["start_yaw"]),
                ],
                dtype=float,
            )
            nominal = matrix(first, "nominal")
            inertia = matrix(first, "inertia")
            idot_filtered = matrix(first, "idot_filtered")
            idot_raw = matrix(first, "idot_raw")
            omegas = {model: start_omega.copy() for model in MODELS}
            rpys = {model: start_rpy.copy() for model in MODELS}

            stop = min(len(rows), start + max_horizon)
            elapsed_time = 0.0
            for index in range(start, stop):
                if index > start:
                    previous = rows[index - 1]
                    current = rows[index]
                    expected = float(previous["time"]) + float(current["dt"])
                    if abs(float(current["time"]) - expected) > 5.0e-4:
                        break
                row = rows[index]
                horizon = index - start + 1
                dt = float(row["dt"])
                angular_impulse = np.array(
                    [
                        float(row["moment_impulse_x"]),
                        float(row["moment_impulse_y"]),
                        float(row["moment_impulse_z"]),
                    ],
                    dtype=float,
                )
                actual = vector(row, "actual")

                for model in MODELS:
                    omega_before = omegas[model].copy()
                    if model == "SRBM":
                        rotation = rz(rpys[model][2])
                        inertia_step = rotation @ nominal @ rotation.T
                        idot_step = np.zeros((3, 3))
                    elif model == "VI-frozen":
                        inertia_step = inertia
                        idot_step = np.zeros((3, 3))
                    elif model == "IR-frozen":
                        inertia_step = inertia
                        idot_step = idot_filtered
                    elif model == "IR-linear":
                        inertia_step = inertia + elapsed_time * idot_filtered
                        inertia_step = 0.5 * (inertia_step + inertia_step.T)
                        idot_step = idot_filtered
                    elif model == "IR-NF-frozen":
                        inertia_step = inertia
                        idot_step = idot_raw
                    else:
                        raise ValueError(model)

                    omegas[model] = omega_before + np.linalg.solve(
                        inertia_step,
                        angular_impulse - idot_step @ omega_before * dt,
                    )
                    rpys[model] += dt * (rz(rpys[model][2]).T @ omega_before)
                    errors[(horizon, model)].append(actual - omegas[model])
                elapsed_time += dt

    summary: list[dict[str, object]] = []
    for horizon in range(1, max_horizon + 1):
        for model in MODELS:
            values = np.asarray(errors[(horizon, model)])
            norms = np.linalg.norm(values, axis=1)
            summary.append(
                {
                    "horizon_steps": horizon,
                    "horizon_ms": 5 * horizon,
                    "model": model,
                    "samples": len(values),
                    "omega_rmse": float(np.sqrt(np.mean(np.sum(values * values, axis=1)))),
                    "wz_rmse": float(np.sqrt(np.mean(values[:, 2] ** 2))),
                    "omega_error_p95": float(np.percentile(norms, 95)),
                }
            )
    return summary, errors


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        x = [float(row["horizon_ms"]) for row in selected]
        axes[0].plot(
            x,
            [float(row["omega_rmse"]) for row in selected],
            marker="o",
            linewidth=1.5,
            markersize=3.8,
            color=COLORS[model],
            label=model,
        )
        axes[1].plot(
            x,
            [float(row["wz_rmse"]) for row in selected],
            marker="o",
            linewidth=1.5,
            markersize=3.8,
            color=COLORS[model],
            label=model,
        )
    axes[0].set_ylabel(r"Angular-velocity RMSE (rad/s)")
    axes[1].set_ylabel(r"$\omega_z$ RMSE (rad/s)")
    for axis in axes:
        axis.set_xlabel("Prediction horizon (ms)")
        axis.set_xticks(range(5, 5 * max(row["horizon_steps"] for row in summary) + 1, 5))
        axis.grid(True, alpha=0.35)
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv", nargs="+", type=Path)
    parser.add_argument("--start-time", type=float, default=4.0)
    parser.add_argument("--end-time", type=float, default=math.inf)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.prediction_csv[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    trials = read_rows(args.prediction_csv, args.start_time, args.end_time)
    summary, _ = calculate(trials, args.max_horizon)
    summary_path = output_dir / "ten_step_prediction_summary.csv"
    figure_path = output_dir / "ten_step_prediction_error.png"
    write_summary(summary_path, summary)
    plot_summary(figure_path, summary)

    horizon = args.max_horizon
    print(f"H={horizon} ({5 * horizon} ms)")
    for row in summary:
        if row["horizon_steps"] == horizon:
            print(
                f"  {row['model']:<14} "
                f"omega={float(row['omega_rmse']):.6f} "
                f"wz={float(row['wz_rmse']):.6f} "
                f"p95={float(row['omega_error_p95']):.6f}"
            )
    print(summary_path)
    print(figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
