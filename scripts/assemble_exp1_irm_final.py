#!/usr/bin/env python3
"""Assemble matched Experiment 1 data for the selected controller comparison."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRBM_DIR = ROOT / "record" / "lambda_filter_turn_exp1_20260729_093310"
DEFAULT_OUT_DIR = ROOT / "record" / "exp1_three_model_final_tau001_reset_20260805"
LAMBDAS = tuple(round(1.0 + 0.1 * index, 1) for index in range(13))
REPEATS = range(1, 6)
START_TIME = 4.0
HORIZON_STEPS = 10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
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


def tracking_rmse(path: Path, terminal_time: float) -> tuple[float, float]:
    sum_wz = 0.0
    sum_vx = 0.0
    count = 0
    for row in read_csv(path):
        time = float(row["time"])
        if time < START_TIME:
            continue
        if time > terminal_time:
            break
        wz_error = float(row["wz"]) - float(row["wz_ref"])
        vx_error = float(row["vx"]) - float(row["vx_ref"])
        if math.isfinite(wz_error) and math.isfinite(vx_error):
            sum_wz += wz_error * wz_error
            sum_vx += vx_error * vx_error
            count += 1
    if count == 0:
        raise RuntimeError(f"No tracking samples in {path}")
    return math.sqrt(sum_wz / count), math.sqrt(sum_vx / count)


def prediction_rmse(path: Path, terminal_time: float) -> float:
    sum_sq = 0.0
    count = 0
    for row in read_csv(path):
        if int(row["horizon_steps"]) != HORIZON_STEPS:
            continue
        origin = float(row["origin_time"])
        target = float(row["target_time"])
        if origin < START_TIME or target > terminal_time:
            continue
        error = float(row["err_wz"])
        if math.isfinite(error):
            sum_sq += error * error
            count += 1
    if count == 0:
        raise RuntimeError(f"No ten-step prediction samples in {path}")
    return math.sqrt(sum_sq / count)


def validate_irm_metadata(path: Path) -> int:
    rows = read_csv(path / "metadata.csv")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metadata row in {path}")
    row = rows[0]
    if row.get("controller") != "ir_cmpc_hrel":
        raise RuntimeError(f"Unexpected controller in {path}")
    if not math.isclose(float(row["ig_dot_filter_tau"]), 0.01):
        raise RuntimeError(f"Unexpected inertia-rate filter in {path}")
    if not math.isclose(float(row["hrel_dot_filter_tau"]), 0.01):
        raise RuntimeError(f"Unexpected relative-momentum-rate filter in {path}")
    if row.get("hrel_contact_reset", "").lower() not in {"true", "1"}:
        raise RuntimeError(f"Contact reset is not enabled in {path}")
    return int(row["uncertainty_rep"])


def source_file(directory: Path, row: dict[str, str], key: str) -> Path:
    path = directory / row[key]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srbm-dir", type=Path, default=DEFAULT_SRBM_DIR)
    parser.add_argument("--irm-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--omit-ir",
        action="store_true",
        help="Assemble only the paired SRBM and IRM-CMPC comparison.",
    )
    args = parser.parse_args()

    srbm_dir = args.srbm_dir.resolve()
    out_dir = args.out_dir.resolve()
    irm_index: dict[tuple[float, int], tuple[dict[str, str], Path]] = {}
    provenance: list[dict[str, object]] = []
    for raw_directory in args.irm_dir:
        directory = raw_directory.resolve()
        rep = validate_irm_metadata(directory)
        for row in read_csv(directory / "summary.csv"):
            lambda_scale = round(float(row["lambda_scale"]), 1)
            if lambda_scale not in LAMBDAS:
                continue
            key = (lambda_scale, rep)
            if key in irm_index:
                raise RuntimeError(f"Duplicate IRM-CMPC sample {key}")
            irm_index[key] = (row, directory)
            provenance.append(
                {
                    "controller": "irm_cmpc",
                    "lambda_scale": lambda_scale,
                    "rep": rep,
                    "source_dir": directory.relative_to(ROOT),
                }
            )

    expected = {(lambda_scale, rep) for lambda_scale in LAMBDAS for rep in REPEATS}
    missing = sorted(expected - set(irm_index))
    if missing:
        raise RuntimeError(f"Missing IRM-CMPC samples: {missing}")

    srbm_rows = [
        row
        for row in read_csv(srbm_dir / "trials.csv")
        if row["controller"] == "srbm"
        and round(float(row["lambda_scale"]), 1) in LAMBDAS
    ]
    srbm_index = {
        (round(float(row["lambda_scale"]), 1), int(row["rep"])): row
        for row in srbm_rows
    }
    if set(srbm_index) != expected:
        raise RuntimeError("SRBM source does not contain the complete matched sweep")
    ir_rows = [
        row
        for row in read_csv(srbm_dir / "trials.csv")
        if row["controller"] == "vicm"
        and round(float(row["lambda_scale"]), 1) in LAMBDAS
    ]
    ir_index = {
        (round(float(row["lambda_scale"]), 1), int(row["rep"])): row
        for row in ir_rows
    }
    if set(ir_index) != expected:
        raise RuntimeError("IR-CMPC source does not contain the complete matched sweep")

    trial_rows: list[dict[str, object]] = []
    data_dir = out_dir / "csv"
    data_dir.mkdir(parents=True, exist_ok=True)
    for lambda_scale in LAMBDAS:
        for rep in REPEATS:
            srbm = srbm_index[(lambda_scale, rep)]
            irm, irm_dir = irm_index[(lambda_scale, rep)]
            ir = ir_index[(lambda_scale, rep)]
            common_times = [float(srbm["final_time"]), float(irm["final_time"])]
            if not args.omit_ir:
                common_times.append(float(ir["final_time"]))
            common_end = min(common_times)

            srbm_trace = srbm_dir / srbm["trace_path"]
            srbm_horizon = srbm_dir / f"{srbm['case']}_mpc_horizon.csv"
            ir_trace = srbm_dir / ir["trace_path"]
            ir_horizon = srbm_dir / f"{ir['case']}_mpc_horizon.csv"
            irm_trace = source_file(irm_dir, irm, "trace_path")
            irm_horizon = source_file(irm_dir, irm, "horizon_path")

            controller_sources = [
                (
                    "srbm",
                    "SRBM",
                    float(srbm["final_time"]),
                    srbm_trace,
                    srbm_horizon,
                    srbm_dir,
                ),
                (
                    "irm_cmpc",
                    "IRM-CMPC",
                    float(irm["final_time"]),
                    irm_trace,
                    irm_horizon,
                    irm_dir,
                ),
            ]
            if not args.omit_ir:
                controller_sources.insert(
                    1,
                    (
                        "ir_cmpc",
                        "IR-CMPC",
                        float(ir["final_time"]),
                        ir_trace,
                        ir_horizon,
                        srbm_dir,
                    ),
                )
            for controller, label, final_time, trace, horizon, source in controller_sources:
                stem = f"lam{lambda_scale:.1f}_{controller}_r{rep}"
                trace_copy = data_dir / f"{stem}_trace.csv"
                horizon_copy = data_dir / f"{stem}_mpc_horizon.csv"
                copy_if_needed(trace, trace_copy)
                copy_if_needed(horizon, horizon_copy)
                wz_tracking, vx_tracking = tracking_rmse(trace, common_end)
                h10 = prediction_rmse(horizon, final_time)
                trial_rows.append(
                    {
                        "lambda_scale": lambda_scale,
                        "rep": rep,
                        "controller": controller,
                        "controller_label": label,
                        "final_time": final_time,
                        "common_terminal_time": common_end,
                        "h10_wz_prediction_rmse_own_trajectory": h10,
                        "wz_tracking_rmse_common_window": wz_tracking,
                        "vx_tracking_rmse_common_window": vx_tracking,
                        "trace_path": trace_copy.relative_to(out_dir),
                        "horizon_path": horizon_copy.relative_to(out_dir),
                        "source_dir": source.relative_to(ROOT),
                    }
                )

    summary_rows: list[dict[str, object]] = []
    for lambda_scale in LAMBDAS:
        controllers = [("srbm", "SRBM"), ("irm_cmpc", "IRM-CMPC")]
        if not args.omit_ir:
            controllers.insert(1, ("ir_cmpc", "IR-CMPC"))
        for controller, label in controllers:
            group = [
                row
                for row in trial_rows
                if row["controller"] == controller
                and math.isclose(float(row["lambda_scale"]), lambda_scale)
            ]
            survival = [float(row["final_time"]) for row in group]
            h10 = [float(row["h10_wz_prediction_rmse_own_trajectory"]) for row in group]
            wz = [float(row["wz_tracking_rmse_common_window"]) for row in group]
            vx = [float(row["vx_tracking_rmse_common_window"]) for row in group]
            summary_rows.append(
                {
                    "lambda_scale": lambda_scale,
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
                    "mean_vx_tracking_rmse": statistics.mean(vx),
                    "sample_sd_vx_tracking_rmse": sample_sd(vx),
                }
            )

    provenance.append(
        {
            "controller": "srbm",
            "lambda_scale": "1.0--2.2",
            "rep": "1--5",
            "source_dir": srbm_dir.relative_to(ROOT),
        }
    )
    if not args.omit_ir:
        provenance.append(
            {
                "controller": "ir_cmpc",
                "lambda_scale": "1.0--2.2",
                "rep": "1--5",
                "source_dir": srbm_dir.relative_to(ROOT),
            }
        )
    write_csv(out_dir / "trials.csv", trial_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "provenance.csv", provenance)
    write_csv(
        out_dir / "experiment_parameters.csv",
        [
            {
                "lambda_values": "1.0:0.1:2.2",
                "repeats": 5,
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
                "torque_bias_scale": 1.0,
                "torque_bias_clamp": "disabled",
                "mpc_state_weights": "50 50 80 1 200 1 1 1 10 100 10 1",
                "rep_1": "no injected noise or push",
                "reps_2_to_5": "paired light measurement noise and 20 N, 0.10 s horizontal push",
            }
        ],
    )
    uncertainty_source = srbm_dir / "uncertainty_profiles.csv"
    if uncertainty_source.exists():
        write_csv(out_dir / "uncertainty_profiles.csv", read_csv(uncertainty_source))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
