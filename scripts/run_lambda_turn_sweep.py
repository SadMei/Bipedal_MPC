#!/usr/bin/env python3
"""Sequential lambda sweep for SRBM/VICM straight and turning trials."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
BIN = BUILD_DIR / "walk_mpc_wbc"
RECORD_DIR = REPO_ROOT / "record"


def token(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def rms(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(v * v for v in values) / len(values))


@dataclass
class TrialResult:
    case: str
    controller: str
    condition: str
    lambda_scale: float
    rep: int
    fall: int
    fall_time: float
    final_time: float
    steps: int
    rms_wz_err: float
    max_abs_wz_err: float
    rms_yaw_err: float
    max_abs_yaw_err: float
    rms_srbm_pred_err: float
    rms_vicm_pred_err: float
    controller_mass: float
    controller_leg_mass: float


def parse_trial(
    case: str,
    controller: str,
    condition: str,
    lambda_scale: float,
    rep: int,
    trace_path: Path,
    pred_path: Path,
    sim_end: float,
) -> TrialResult:
    with trace_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty trace for {case}")

    final_time = float(rows[-1]["time"])
    fall = int(float(rows[-1]["fall_detected"]))
    fall_time = final_time if fall else sim_end
    steps = int(float(rows[-1]["step_count"]))
    controller_mass = float(rows[-1]["controller_mass"])
    controller_leg_mass = float(rows[-1]["controller_leg_mass"])

    eval_rows = [r for r in rows if float(r["time"]) >= 4.0]
    wz_err = [float(r["wz"]) - float(r["wz_ref"]) for r in eval_rows]
    yaw_err = [float(r["yaw"]) - float(r["yaw_ref"]) for r in eval_rows]

    srbm_pred_err: list[float] = []
    vicm_pred_err: list[float] = []
    if pred_path.exists():
      with pred_path.open(newline="") as f:
          for row in csv.DictReader(f):
              if float(row["time"]) >= 4.0:
                  srbm_pred_err.append(float(row["srbm_err_norm"]))
                  vicm_pred_err.append(float(row["vicm_err_norm"]))

    return TrialResult(
        case=case,
        controller=controller,
        condition=condition,
        lambda_scale=lambda_scale,
        rep=rep,
        fall=fall,
        fall_time=fall_time,
        final_time=final_time,
        steps=steps,
        rms_wz_err=rms(wz_err),
        max_abs_wz_err=max((abs(v) for v in wz_err), default=math.nan),
        rms_yaw_err=rms(yaw_err),
        max_abs_yaw_err=max((abs(v) for v in yaw_err), default=math.nan),
        rms_srbm_pred_err=rms(srbm_pred_err),
        rms_vicm_pred_err=rms(vicm_pred_err),
        controller_mass=controller_mass,
        controller_leg_mass=controller_leg_mass,
    )


def write_result(writer: csv.DictWriter, result: TrialResult) -> None:
    row = result.__dict__.copy()
    writer.writerow(row)


def summarize_lambda(results: list[TrialResult]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str, str], list[TrialResult]] = {}
    for result in results:
        key = (result.lambda_scale, result.controller, result.condition)
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, object]] = []
    for (lambda_scale, controller, condition), group in sorted(groups.items()):
        final_times = [g.final_time for g in group]
        falls = [g.fall for g in group]
        rows.append(
            {
                "lambda_scale": lambda_scale,
                "controller": controller,
                "condition": condition,
                "n": len(group),
                "fall_count": sum(falls),
                "mean_final_time": sum(final_times) / len(final_times),
                "std_final_time": math.sqrt(
                    sum((t - sum(final_times) / len(final_times)) ** 2
                        for t in final_times) / len(final_times)
                ),
                "min_final_time": min(final_times),
                "max_final_time": max(final_times),
                "mean_rms_wz_err": sum(g.rms_wz_err for g in group) / len(group),
                "mean_rms_yaw_err": sum(g.rms_yaw_err for g in group) / len(group),
                "mean_rms_srbm_pred_err": sum(g.rms_srbm_pred_err for g in group) / len(group),
                "mean_rms_vicm_pred_err": sum(g.rms_vicm_pred_err for g in group) / len(group),
            }
        )
    return rows


def run_trial(
    out_dir: Path,
    lambda_scale: float,
    controller: str,
    condition: str,
    rep: int,
    sim_end: float,
) -> TrialResult:
    case = f"lam{token(lambda_scale)}_{controller}_{condition}_r{rep}"
    env = os.environ.copy()
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_EXP": "1",
            "ODC_RUN_LABEL": case,
            "ODC_USE_LEG_LAMBDA_SCALE": "1",
            "ODC_LEG_LAMBDA_SCALE": f"{lambda_scale:.12g}",
            "ODC_TARGET_SPEED_X": "1.5",
            "ODC_TARGET_SPEED_Y": "0",
            "ODC_TSWING": "0.45",
            "ODC_GAIT_SWITCH_FORCE_SOURCE": "touch",
            "ODC_SIM_END_TIME": f"{sim_end:.12g}",
            "ODC_TAU_BIAS_SCALE": "0.5",
            "ODC_PREDICT_IG_LINEAR": "0",
            "ODC_LINEAR_TAU_DYNAMICS": "1" if controller == "vicm" else "0",
            "ODC_MPC_L_DIAG": "50 50 20 1 200 1 1 1 2 100 10 1",
            "ODC_TORQUE_LIMIT_SCALE": "1.2",
            "ODC_WALK_LEG_PD_SCALE": "1.2",
            "ODC_LOG_PREDICTION_ERROR": "1",
            "ODC_PRINT_MPC_TIMING": "0",
            "ODC_PRINT_FR_FF": "0",
        }
    )
    if controller == "vicm":
        env["ODC_USE_VICM"] = "1"
        env["ODC_USE_TAU_BIAS"] = "1"
    else:
        env["ODC_USE_VICM"] = "0"
        env["ODC_USE_TAU_BIAS"] = "0"

    if condition == "turn":
        env.update(
            {
                "ODC_SINE_TURN": "1",
                "ODC_SINE_WZ_BASE": "0",
                "ODC_SINE_WZ_AMP": "0.25",
                "ODC_SINE_WZ_PERIOD": "4.0",
                "ODC_SINE_WZ_START_TIME": "4.0",
            }
        )
    else:
        env["ODC_SINE_TURN"] = "0"

    log_path = out_dir / f"{case}.log"
    with log_path.open("w") as log:
        subprocess.run(
            [str(BIN)],
            cwd=BUILD_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    trace_src = RECORD_DIR / "exp1_trace.csv"
    trace_dst = out_dir / f"{case}_trace.csv"
    shutil.copyfile(trace_src, trace_dst)

    pred_src = RECORD_DIR / f"pred_error_exp1_{case}_lf0.500000.csv"
    pred_dst = out_dir / f"{case}_pred_error.csv"
    if pred_src.exists():
        shutil.copyfile(pred_src, pred_dst)

    return parse_trial(
        case, controller, condition, lambda_scale, rep, trace_dst, pred_dst, sim_end
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--stop-threshold", type=float, default=8.0)
    args = parser.parse_args()

    if not BIN.exists():
        raise FileNotFoundError(f"missing executable: {BIN}")

    out_dir = RECORD_DIR / f"lambda_turn_sweep_{os.environ.get('ODC_RUN_STAMP', '')}".rstrip("_")
    if out_dir.exists():
        suffix = 1
        while (RECORD_DIR / f"{out_dir.name}_{suffix}").exists():
            suffix += 1
        out_dir = RECORD_DIR / f"{out_dir.name}_{suffix}"
    out_dir.mkdir(parents=True)

    result_fields = list(TrialResult.__dataclass_fields__.keys())
    all_results: list[TrialResult] = []
    results_path = out_dir / "trials.csv"
    summary_path = out_dir / "summary.csv"

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()

        lambda_scale = args.start
        while lambda_scale <= args.max_lambda + 1e-9:
            lambda_results: list[TrialResult] = []
            print(f"=== lambda={lambda_scale:.3f} ===", flush=True)
            for condition in ("straight", "turn"):
                for controller in ("vicm", "srbm"):
                    for rep in range(1, args.repeats + 1):
                        result = run_trial(
                            out_dir,
                            lambda_scale,
                            controller,
                            condition,
                            rep,
                            args.sim_end,
                        )
                        write_result(writer, result)
                        f.flush()
                        lambda_results.append(result)
                        all_results.append(result)
                        print(
                            f"{result.case}: final={result.final_time:.3f}s "
                            f"fall={result.fall} wz_rms={result.rms_wz_err:.3f} "
                            f"yaw_rms={result.rms_yaw_err:.3f}",
                            flush=True,
                        )

            summary_rows = summarize_lambda(all_results)
            if summary_rows:
                with summary_path.open("w", newline="") as sf:
                    sw = csv.DictWriter(sf, fieldnames=list(summary_rows[0].keys()))
                    sw.writeheader()
                    sw.writerows(summary_rows)

            vicm_results = [r for r in lambda_results if r.controller == "vicm"]
            vicm_unusable = (
                len(vicm_results) == 2 * args.repeats
                and all(r.fall for r in vicm_results)
                and max(r.final_time for r in vicm_results) < args.stop_threshold
            )
            if vicm_unusable:
                print(
                    f"Stop: VICM unusable at lambda={lambda_scale:.3f} "
                    f"(all falls, max final < {args.stop_threshold}s).",
                    flush=True,
                )
                break

            lambda_scale = round(lambda_scale + args.step, 10)

    print(f"OUT={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
