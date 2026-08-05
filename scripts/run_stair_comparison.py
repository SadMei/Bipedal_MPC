#!/usr/bin/env python3
"""Run matched SRBM/IRM-CMPC staircase simulations and summarize them."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build" / "walk_mpc_wbc"
RECORD_DIR = ROOT / "record"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--models", nargs="+", choices=("srbm", "irm"),
                        default=("srbm", "irm"))
    parser.add_argument("--lambdas", nargs="+", type=float, default=(1.0, 1.8))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sim-time", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--swing-time", type=float, default=0.55)
    parser.add_argument("--step-height", type=float, default=0.40)
    parser.add_argument("--min-touchdown-phase", type=float, default=0.85)
    parser.add_argument("--label-prefix", default="stair_compare")
    parser.add_argument("--summary", type=Path,
                        default=RECORD_DIR / "stair_comparison_summary.csv")
    return parser.parse_args()


def common_environment(args: argparse.Namespace, leg_lambda: float,
                       label: str) -> dict[str, str]:
    values = {
        "ODC_HEADLESS": "1",
        "ODC_SCENE_XML": "../models/scene_staircase_015.xml",
        "ODC_STAIR_MODE": "1",
        "ODC_STAIR_CONTACT_PREVIEW": "1",
        "ODC_STAIR_FIRST_RISER_X": "0.5",
        "ODC_STAIR_TREAD_DEPTH": "0.5",
        "ODC_STAIR_RISER_HEIGHT": "0.15",
        "ODC_STAIR_MAX_HEIGHT": "1.5",
        "ODC_STAIR_LANDING_MARGIN": "0.10",
        "ODC_GAIT_MIN_TOUCHDOWN_PHASE": f"{args.min_touchdown_phase:.12g}",
        "ODC_GAIT_TOUCHDOWN_POSITION_GATE": "1",
        "ODC_GAIT_TOUCHDOWN_POSITION_TOLERANCE": "0.18",
        "ODC_GAIT_TOUCHDOWN_HEIGHT_TOLERANCE": "0.15",
        "ODC_FOOT_STEP_HEIGHT": f"{args.step_height:.12g}",
        "ODC_TARGET_SPEED_X": f"{args.speed:.12g}",
        "ODC_TARGET_SPEED_Y": "0",
        "ODC_TSWING": f"{args.swing_time:.12g}",
        "ODC_GAIT_SWITCH_FORCE_THRESHOLD": "100",
        "ODC_USE_LEG_LAMBDA_SCALE": "1",
        "ODC_LEG_LAMBDA_SCALE": f"{leg_lambda:.12g}",
        "ODC_WBC_POSROT_POS_KP_SCALE": "1",
        "ODC_WBC_POSROT_POS_KD_SCALE": "1",
        "ODC_WBC_POSROT_ATT_KP_SCALE": "0.35",
        "ODC_WBC_POSROT_ATT_KD_SCALE": "0.35",
        "ODC_TORQUE_LIMIT_SCALE": "1.2",
        "ODC_WALK_LEG_PD_SCALE": "1.2",
        "ODC_SENSOR_NOISE_ENABLE": "0",
        "ODC_PUSH_FORCE": "0",
        "ODC_SINE_TURN": "0",
        "ODC_SIM_END_TIME": f"{args.sim_time:.12g}",
        "ODC_LOG_PREDICTION_ERROR": "1",
        "ODC_ISOLATE_EXPERIMENT_OUTPUTS": "1",
        "ODC_PRINT_FR_FF": "0",
        "ODC_PRINT_MPC_TIMING": "0",
        "ODC_RUN_LABEL": label,
    }
    env = os.environ.copy()
    env.update(values)
    return env


def model_environment(model: str) -> dict[str, str]:
    if model == "srbm":
        return {
            "ODC_USE_VICM": "0",
            "ODC_USE_TAU_BIAS": "0",
            "ODC_LINEAR_TAU_DYNAMICS": "0",
            "ODC_USE_HREL_RATE": "0",
        }
    return {
        "ODC_USE_VICM": "1",
        "ODC_USE_TAU_BIAS": "1",
        "ODC_LINEAR_TAU_DYNAMICS": "1",
        "ODC_USE_HREL_RATE": "1",
        "ODC_IG_DOT_FILTER_TAU": "0.01",
        "ODC_HREL_DOT_FILTER_TAU": "0.01",
        "ODC_HREL_RESET_ON_CONTACT_SWITCH": "1",
        # Stair impacts excite finite-difference spikes. This bound is shared by
        # the inertia-rate and relative-momentum-rate moment corrections.
        "ODC_TAU_NON_NORM_LIMIT": "60",
    }


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rms(values: Iterable[float]) -> float:
    data = list(values)
    return math.sqrt(sum(value * value for value in data) / len(data)) if data else math.nan


def summarize(label: str, model: str, leg_lambda: float,
              sim_time: float) -> dict[str, object]:
    trace_path = RECORD_DIR / f"exp1_{label}_trace.csv"
    traces = read_rows(trace_path)
    final = traces[-1]
    evaluation = [row for row in traces if float(row["time"]) >= 4.0]
    horizon_candidates = sorted(RECORD_DIR.glob(f"mpc_horizon_exp1_{label}_lf*.csv"))
    horizon_rows = read_rows(horizon_candidates[-1]) if horizon_candidates else []
    horizon_ten = [row for row in horizon_rows
                   if int(float(row["horizon_steps"])) == 10
                   and float(row["origin_time"]) >= 4.0]
    return {
        "label": label,
        "model": "IRM-CMPC" if model == "irm" else "SRBM",
        "leg_lambda": leg_lambda,
        "command_speed_x": float(final["target_speed_x"]),
        "final_time": float(final["time"]),
        "completed_simulation": int(float(final["time"]) >= sim_time - 0.01),
        "step_count": int(float(final["step_count"])),
        "max_base_x": max(float(row["base_x"]) for row in traces),
        "max_terrain_height": max(float(row["terrain_height"]) for row in traces),
        "vx_tracking_rms": rms(float(row["vx"]) - float(row["vx_ref"])
                               for row in evaluation),
        "wz_stabilization_rms": rms(float(row["wz"]) - float(row["wz_ref"])
                                    for row in evaluation),
        "wz_prediction_rms_h10": rms(float(row["err_wz"])
                                     for row in horizon_ten),
        "h10_samples": len(horizon_ten),
    }


def main() -> None:
    args = parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"simulation binary not found: {binary}")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for leg_lambda in args.lambdas:
        for model in args.models:
            for repeat in range(1, args.repeats + 1):
                lambda_tag = str(leg_lambda).replace(".", "p")
                label = f"{args.label_prefix}_{model}_l{lambda_tag}_r{repeat}"
                env = common_environment(args, leg_lambda, label)
                env.update(model_environment(model))
                log_path = RECORD_DIR / f"{label}.log"
                print(f"[run] model={model} lambda={leg_lambda:g} repeat={repeat} "
                      f"label={label}", flush=True)
                with log_path.open("w") as log:
                    subprocess.run([str(binary)], cwd=binary.parent, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   check=True)
                result = summarize(label, model, leg_lambda, args.sim_time)
                results.append(result)
                print("[result] " + " ".join(
                    f"{key}={value}" for key, value in result.items()
                    if key in ("final_time", "step_count", "max_terrain_height",
                               "vx_tracking_rms", "wz_prediction_rms_h10")),
                    flush=True)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"[summary] {args.summary.resolve()}")


if __name__ == "__main__":
    main()
