#!/usr/bin/env python3
"""Post-process full push-recovery traces without selecting one favorable cutoff."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def per_step_metrics(trace_path: Path, window_s: float, max_steps: int) -> list[dict[str, float | int]]:
    rows = read_csv(trace_path)
    trigger = next((row for row in rows if value(row, "push_triggered", 0.0) > 0.5), None)
    if trigger is None:
        return []
    trigger_time = value(trigger, "push_actual_start")
    metrics: list[dict[str, float | int]] = []
    for step in range(1, max_steps + 1):
        endpoint = next(
            (
                row
                for row in rows
                if value(row, "time") >= trigger_time
                and int(value(row, "recovery_steps", 0.0)) >= step
            ),
            None,
        )
        if endpoint is None:
            continue
        endpoint_time = value(endpoint, "time")
        window = [
            row
            for row in rows
            if endpoint_time - window_s <= value(row, "time") <= endpoint_time
        ]
        mean_vx = mean([value(row, "vx") for row in window])
        mean_vy = mean([value(row, "vy") for row in window])
        metrics.append(
            {
                "step": step,
                "time": endpoint_time,
                "planar_velocity": math.hypot(mean_vx, mean_vy),
                "max_torso": max(value(row, "torso_angle_error") for row in window),
                "upright": int(all(value(row, "fall_detected", 0.0) < 0.5 for row in window)),
            }
        )
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_recovery_map(
    per_step_rows: list[dict[str, object]],
    *,
    horizon: int,
    speed_threshold: float,
    torso_threshold: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float, int], list[dict[str, object]]] = {}
    for row in per_step_rows:
        key = (
            str(row["controller_label"]),
            float(row["push_angle_deg"]),
            float(row["push_force"]),
            int(float(row["rep"])),
        )
        grouped.setdefault(key, []).append(row)
    selected: list[dict[str, object]] = []
    for (controller, angle, force, rep), metrics in sorted(grouped.items()):
        recovered = any(
            int(metric["step"]) <= horizon
            and int(metric["upright"]) == 1
            and float(metric["planar_velocity"]) <= speed_threshold
            and float(metric["max_torso"]) <= torso_threshold
            for metric in metrics
        )
        selected.append(
            {
                "controller_label": controller,
                "push_angle_deg": angle,
                "push_force": force,
                "rep": rep,
                "recovered": int(recovered),
            }
        )
    return selected


def plot_selected_map(result_dir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    controllers = [name for name in ("SRBM", "IR-CMPC") if any(row["controller_label"] == name for row in rows)]
    directions = sorted({float(row["push_angle_deg"]) for row in rows})
    forces = sorted({float(row["push_force"]) for row in rows})
    fig, axes = plt.subplots(1, len(controllers), figsize=(7.0, 3.15), sharey=True)
    axes = np.atleast_1d(axes)
    cmap = ListedColormap(["#FFFFFF", "#2F67B2"])
    for ax, controller in zip(axes, controllers):
        matrix = np.zeros((len(directions), len(forces)))
        for row in rows:
            if row["controller_label"] != controller:
                continue
            i = directions.index(float(row["push_angle_deg"]))
            j = forces.index(float(row["push_force"]))
            matrix[i, j] = max(matrix[i, j], int(row["recovered"]))
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
        recovered_count = int(matrix.sum())
        ax.set_title(f"{controller} ({recovered_count}/{matrix.size})", fontsize=9)
        ax.set_xlabel("Push force (N)", fontsize=8)
        ax.set_xticks(range(len(forces)), [f"{force:.0f}" for force in forces], rotation=45, fontsize=7)
        ax.set_yticks(range(len(directions)), [f"{direction:.0f}" for direction in directions], fontsize=7)
        ax.set_xticks(np.arange(-0.5, len(forces), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(directions), 1), minor=True)
        ax.grid(which="minor", color="#C9CED8", linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Push direction (deg)", fontsize=8)
    fig.legend(
        handles=[
            Patch(facecolor="#2F67B2", edgecolor="#2F67B2", label="Recovered"),
            Patch(facecolor="#FFFFFF", edgecolor="#8A9099", label="Not recovered"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(result_dir / "selected_three_step_recovery_map.png", dpi=300)
    fig.savefig(result_dir / "selected_three_step_recovery_map.pdf")
    plt.close(fig)


def continuous_horizon_metrics(
    per_step_rows: list[dict[str, object]],
    trials: list[dict[str, str]],
    horizon: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, float, int], list[dict[str, object]]] = {}
    for row in per_step_rows:
        if int(float(row["step"])) > horizon or int(float(row["upright"])) != 1:
            continue
        key = (
            str(row["controller_label"]),
            float(row["push_angle_deg"]),
            float(row["push_force"]),
            int(float(row["rep"])),
        )
        grouped.setdefault(key, []).append(row)

    trial_by_key = {
        (
            trial["controller_label"],
            float(trial["push_angle_deg"]),
            float(trial["push_force"]),
            int(float(trial["rep"])),
        ): trial
        for trial in trials
    }
    rows: list[dict[str, object]] = []
    for key, trial in sorted(trial_by_key.items()):
        metrics = grouped.get(key, [])
        rows.append(
            {
                "controller_label": key[0],
                "push_angle_deg": key[1],
                "push_force": key[2],
                "rep": key[3],
                "horizon_steps": horizon,
                "min_planar_velocity": min(
                    (float(metric["planar_velocity"]) for metric in metrics),
                    default=math.nan,
                ),
                "fall": int(float(trial["fall"])),
                "final_time": float(trial["final_time"]),
            }
        )
    return rows


def plot_continuous_horizon_map(
    result_dir: Path,
    rows: list[dict[str, object]],
    horizon: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    controllers = [
        name
        for name in ("SRBM", "IR-CMPC")
        if any(row["controller_label"] == name for row in rows)
    ]
    if controllers != ["SRBM", "IR-CMPC"]:
        return
    directions = sorted({float(row["push_angle_deg"]) for row in rows})
    forces = sorted({float(row["push_force"]) for row in rows})
    matrices: dict[str, np.ndarray] = {}
    falls: dict[str, np.ndarray] = {}
    for controller in controllers:
        matrix = np.full((len(directions), len(forces)), np.nan)
        fall_matrix = np.zeros_like(matrix, dtype=bool)
        for row in rows:
            if row["controller_label"] != controller:
                continue
            i = directions.index(float(row["push_angle_deg"]))
            j = forces.index(float(row["push_force"]))
            matrix[i, j] = float(row["min_planar_velocity"])
            fall_matrix[i, j] = bool(int(row["fall"]))
        matrices[controller] = matrix
        falls[controller] = fall_matrix

    difference = matrices["IR-CMPC"] - matrices["SRBM"]
    finite_values = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in matrices.values()]
    )
    vmax = max(0.5, float(np.nanmax(finite_values)))
    diff_limit = max(0.05, float(np.nanmax(np.abs(difference))))
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), sharey=True)
    for ax, controller in zip(axes[:2], controllers):
        image = ax.imshow(
            matrices[controller],
            cmap="Blues_r",
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
            origin="lower",
        )
        y, x = np.where(falls[controller])
        ax.scatter(x, y, marker="x", color="#111111", s=24, linewidths=1.0)
        ax.set_title(controller, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03, label="Residual speed (m/s)")

    diff_image = axes[2].imshow(
        difference,
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
        aspect="auto",
        origin="lower",
    )
    axes[2].set_title("IR-CMPC minus SRBM", fontsize=9)
    fig.colorbar(diff_image, ax=axes[2], fraction=0.046, pad=0.03, label="Speed difference (m/s)")
    for ax in axes:
        ax.set_xlabel("Push force (N)", fontsize=8)
        ax.set_xticks(
            range(len(forces)),
            [f"{force:.0f}" for force in forces],
            rotation=45,
            fontsize=7,
        )
        ax.set_yticks(
            range(len(directions)),
            [f"{direction:.0f}" for direction in directions],
            fontsize=7,
        )
        ax.set_xticks(np.arange(-0.5, len(forces), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(directions), 1), minor=True)
        ax.grid(which="minor", color="#C9CED8", linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Push direction (deg)", fontsize=8)
    fig.text(
        0.01,
        0.01,
        f"Minimum horizontal speed within {horizon} recovery steps; x marks a later fall.",
        fontsize=8,
        color="#50555E",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(result_dir / f"{horizon}_step_residual_velocity_map.png", dpi=300)
    fig.savefig(result_dir / f"{horizon}_step_residual_velocity_map.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--window", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--speed-thresholds", default="0.20 0.25 0.30 0.35")
    parser.add_argument("--torso-thresholds", default="0.20 0.25 0.30")
    parser.add_argument("--selected-horizon", type=int, default=3)
    parser.add_argument("--selected-speed-threshold", type=float, default=0.25)
    parser.add_argument("--selected-torso-threshold", type=float, default=0.25)
    args = parser.parse_args()

    trials = read_csv(args.result_dir / "trials.csv")
    speed_thresholds = [float(item) for item in args.speed_thresholds.split()]
    torso_thresholds = [float(item) for item in args.torso_thresholds.split()]
    per_step_rows: list[dict[str, object]] = []
    by_case: dict[tuple[str, str], list[dict[str, float | int]]] = {}
    trial_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for trial in trials:
        trace_path = args.result_dir / trial["trace_path"]
        metrics = per_step_metrics(trace_path, args.window, args.max_steps)
        key = (trial["case"], trial["controller_label"])
        by_case[key] = metrics
        trial_lookup[key] = trial
        for metric in metrics:
            per_step_rows.append(
                {
                    "case": trial["case"],
                    "controller_label": trial["controller_label"],
                    "push_angle_deg": trial["push_angle_deg"],
                    "push_force": trial["push_force"],
                    "rep": trial["rep"],
                    **metric,
                }
            )
    write_csv(args.result_dir / "per_step_stop_metrics.csv", per_step_rows)

    sensitivity: list[dict[str, object]] = []
    controllers = sorted({trial["controller_label"] for trial in trials})
    for horizon in range(2, args.max_steps + 1):
        for speed_threshold in speed_thresholds:
            for torso_threshold in torso_thresholds:
                recovered_by_controller: dict[str, dict[tuple[float, float, int], int]] = {}
                for controller in controllers:
                    outcomes: dict[tuple[float, float, int], int] = {}
                    for key, metrics in by_case.items():
                        trial = trial_lookup[key]
                        if trial["controller_label"] != controller:
                            continue
                        recovered = any(
                            int(metric["step"]) <= horizon
                            and int(metric["upright"]) == 1
                            and float(metric["planar_velocity"]) <= speed_threshold
                            and float(metric["max_torso"]) <= torso_threshold
                            for metric in metrics
                        )
                        case_key = (
                            float(trial["push_angle_deg"]),
                            float(trial["push_force"]),
                            int(float(trial["rep"])),
                        )
                        outcomes[case_key] = int(recovered)
                    recovered_by_controller[controller] = outcomes

                baseline_valid = all(
                    outcome == 1
                    for outcomes in recovered_by_controller.values()
                    for (angle, force, rep), outcome in outcomes.items()
                    if abs(force) < 1e-9
                )
                row: dict[str, object] = {
                    "horizon_steps": horizon,
                    "speed_threshold": speed_threshold,
                    "torso_threshold": torso_threshold,
                    "baseline_valid": int(baseline_valid),
                }
                for controller, outcomes in recovered_by_controller.items():
                    row[f"{controller}_recovered_cells"] = sum(outcomes.values())
                    row[f"{controller}_total_cells"] = len(outcomes)
                if "SRBM" in controllers and "IR-CMPC" in controllers:
                    first, second = "SRBM", "IR-CMPC"
                    common = sorted(
                        set(recovered_by_controller[first])
                        & set(recovered_by_controller[second])
                    )
                    row["ir_cmpc_minus_srbm_cells"] = sum(
                        recovered_by_controller[second][key]
                        - recovered_by_controller[first][key]
                        for key in common
                    )
                sensitivity.append(row)
    write_csv(args.result_dir / "recovery_metric_sensitivity.csv", sensitivity)
    selected = selected_recovery_map(
        per_step_rows,
        horizon=args.selected_horizon,
        speed_threshold=args.selected_speed_threshold,
        torso_threshold=args.selected_torso_threshold,
    )
    write_csv(args.result_dir / "selected_recovery_map.csv", selected)
    plot_selected_map(args.result_dir, selected)
    continuous = continuous_horizon_metrics(
        per_step_rows, trials, args.selected_horizon
    )
    write_csv(
        args.result_dir / f"{args.selected_horizon}_step_residual_velocity.csv",
        continuous,
    )
    plot_continuous_horizon_map(
        args.result_dir, continuous, args.selected_horizon
    )
    print(f"WROTE={args.result_dir / 'per_step_stop_metrics.csv'}")
    print(f"WROTE={args.result_dir / 'recovery_metric_sensitivity.csv'}")
    print(f"WROTE={args.result_dir / 'selected_three_step_recovery_map.png'}")
    print(
        f"WROTE={args.result_dir / f'{args.selected_horizon}_step_residual_velocity_map.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
