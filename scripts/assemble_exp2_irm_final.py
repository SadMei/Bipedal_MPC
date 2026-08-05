#!/usr/bin/env python3
"""Assemble the final four-model Experiment 2 ablation data."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
from pathlib import Path

from vicm_experiment_lib import compute_trace_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "record" / "exp2_final_ablation_lam1p8_20260803_213517"
DEFAULT_IRM = ROOT / "record" / "exp2_final_ablation_lam1p8_20260805_021238"
DEFAULT_OUT = ROOT / "figures" / "experiment_data" / "experiment_2_hrel"
CONTROLLERS = (
    ("srbm", "SRBM"),
    ("vicm_ig", "VI-CMPC"),
    ("ir_cmpc", "IR-CMPC"),
    ("ir_cmpc_hrel", "IRM-CMPC"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_if_needed(source: Path, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, destination)


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--irm-dir", type=Path, default=DEFAULT_IRM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    irm_dir = args.irm_dir.resolve()
    rows = read_csv(base_dir / "trials.csv") + read_csv(irm_dir / "trials.csv")
    trial_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    data_dir = args.out_dir / "csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    for controller, label in CONTROLLERS:
        source = irm_dir if controller == "ir_cmpc_hrel" else base_dir
        group = [row for row in rows if row["controller"] == controller]
        if len(group) != 3:
            raise RuntimeError(f"Expected three {label} trials, found {len(group)}")
        survival: list[float] = []
        h10: list[float] = []
        wz: list[float] = []
        velocity: list[float] = []
        wall: list[float] = []
        for row in group:
            trace = source / row["trace_path"]
            horizon = source / row["horizon_path"]
            metrics = compute_trace_metrics(
                trace,
                horizon,
                sim_end=30.0,
                sine_start=4.0,
                sine_period=4.0,
                max_yaw_rms=999.0,
                max_wz_rms=999.0,
                max_path_rms=999.0,
                max_torso=999.0,
            )
            survival.append(float(row["final_time"]))
            h10.append(float(row["h10_wz_prediction_rmse"]))
            wz.append(float(metrics["rms_wz_err"]))
            velocity.append(float(metrics["rms_vel_err"]))
            wall.append(float(row["mpc_avg_wall_ms"]))
            stem = f"{controller}_r{int(row['rep'])}"
            trace_copy = data_dir / f"{stem}_trace.csv"
            horizon_copy = data_dir / f"{stem}_mpc_horizon.csv"
            copy_if_needed(trace, trace_copy)
            copy_if_needed(horizon, horizon_copy)
            trial_rows.append(
                {
                    "controller": controller,
                    "controller_label": label,
                    "rep": int(row["rep"]),
                    "final_time": float(row["final_time"]),
                    "h10_wz_prediction_rmse": float(row["h10_wz_prediction_rmse"]),
                    "wz_tracking_rmse": float(metrics["rms_wz_err"]),
                    "velocity_tracking_rmse": float(metrics["rms_vel_err"]),
                    "mpc_avg_wall_ms": float(row["mpc_avg_wall_ms"]),
                    "trace_path": trace_copy.relative_to(args.out_dir),
                    "horizon_path": horizon_copy.relative_to(args.out_dir),
                    "source_dir": source.relative_to(ROOT),
                }
            )
        summary_rows.append(
            {
                "controller": controller,
                "controller_label": label,
                "n": len(group),
                "completed_30s": sum(value >= 29.999 for value in survival),
                "mean_survival_time": statistics.mean(survival),
                "sample_sd_survival_time": sample_sd(survival),
                "mean_h10_wz_prediction_rmse": statistics.mean(h10),
                "sample_sd_h10_wz_prediction_rmse": sample_sd(h10),
                "mean_wz_tracking_rmse": statistics.mean(wz),
                "sample_sd_wz_tracking_rmse": sample_sd(wz),
                "mean_velocity_tracking_rmse": statistics.mean(velocity),
                "sample_sd_velocity_tracking_rmse": sample_sd(velocity),
                "mean_mpc_wall_ms": statistics.mean(wall),
                "sample_sd_mpc_wall_ms": sample_sd(wall),
            }
        )

    write_csv(args.out_dir / "trials.csv", trial_rows)
    write_csv(args.out_dir / "summary.csv", summary_rows)
    write_csv(
        args.out_dir / "provenance.csv",
        [
            {
                "controllers": "SRBM, VI-CMPC, IR-CMPC",
                "source_dir": base_dir.relative_to(ROOT),
                "selection": "three matched deterministic runs",
            },
            {
                "controllers": "IRM-CMPC",
                "source_dir": irm_dir.relative_to(ROOT),
                "selection": "tau_h=0.01 s with contact-aware derivative reset",
            },
        ],
    )
    write_csv(
        args.out_dir / "experiment_parameters.csv",
        [
            {
                "lambda_scale": 1.8,
                "repeats": 3,
                "sim_end_s": 30.0,
                "vx_command_mps": 1.5,
                "swing_time_s": 0.45,
                "wz_amplitude_radps": 0.4,
                "wz_period_s": 4.0,
                "wz_start_s": 4.0,
                "wbc_attitude_scale": 0.35,
                "ig_dot_filter_tau_s": 0.01,
                "hrel_dot_filter_tau_s": 0.01,
                "hrel_contact_derivative_reset": True,
                "uncertainty": "none",
            }
        ],
    )
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
