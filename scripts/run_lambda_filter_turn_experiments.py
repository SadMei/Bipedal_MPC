#!/usr/bin/env python3
"""Lambda sweep for filtered VICM turning experiments.

This runner is intentionally narrow: it reproduces the current "default WBC +
filtered Ig_dot" condition used for the lambda-form SRBM/VICM comparison.
It runs VICM first, stops after the high-lambda region is clearly unusable,
then runs SRBM for the completed lambda values.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


@dataclass
class TrialResult:
    case: str
    controller: str
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
    mean_wbc_delta_fr_norm: float
    max_wbc_delta_fr_norm: float
    wall_time_s: float
    log_path: str
    trace_path: str
    pred_path: str


def parse_trial(
    case: str,
    controller: str,
    lambda_scale: float,
    rep: int,
    trace_path: Path,
    pred_path: Path,
    log_path: Path,
    sim_end: float,
    wall_time_s: float,
    out_dir: Path,
) -> TrialResult:
    with trace_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")

    last = rows[-1]
    final_time = float(last["time"])
    fall = int(float(last["fall_detected"]))
    fall_time = final_time if fall else sim_end
    eval_rows = [r for r in rows if float(r["time"]) >= 4.0]
    wz_err = [float(r["wz"]) - float(r["wz_ref"]) for r in eval_rows]
    yaw_err = [float(r["yaw"]) - float(r["yaw_ref"]) for r in eval_rows]

    delta_fr_values: list[float] = []
    for row in eval_rows:
        value = row.get("wbc_delta_fr_norm")
        if value in (None, ""):
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            delta_fr_values.append(parsed)

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
        lambda_scale=lambda_scale,
        rep=rep,
        fall=fall,
        fall_time=fall_time,
        final_time=final_time,
        steps=int(float(last["step_count"])),
        rms_wz_err=rms(wz_err),
        max_abs_wz_err=max((abs(v) for v in wz_err), default=math.nan),
        rms_yaw_err=rms(yaw_err),
        max_abs_yaw_err=max((abs(v) for v in yaw_err), default=math.nan),
        rms_srbm_pred_err=rms(srbm_pred_err),
        rms_vicm_pred_err=rms(vicm_pred_err),
        controller_mass=float(last["controller_mass"]),
        controller_leg_mass=float(last["controller_leg_mass"]),
        mean_wbc_delta_fr_norm=mean(delta_fr_values),
        max_wbc_delta_fr_norm=max(delta_fr_values, default=math.nan),
        wall_time_s=wall_time_s,
        log_path=str(log_path.relative_to(out_dir)),
        trace_path=str(trace_path.relative_to(out_dir)),
        pred_path=str(pred_path.relative_to(out_dir)) if pred_path.exists() else "",
    )


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fields if fields is not None else list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[TrialResult]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str], list[TrialResult]] = {}
    for result in results:
        groups.setdefault((result.lambda_scale, result.controller), []).append(result)

    rows: list[dict[str, object]] = []
    for (lambda_scale, controller), group in sorted(groups.items()):
        times = [g.final_time for g in group]
        wz = [g.rms_wz_err for g in group]
        yaw = [g.rms_yaw_err for g in group]
        pred_s = [g.rms_srbm_pred_err for g in group]
        pred_v = [g.rms_vicm_pred_err for g in group]
        rows.append(
            {
                "lambda_scale": lambda_scale,
                "controller": controller,
                "n": len(group),
                "fall_count": sum(g.fall for g in group),
                "mean_final_time": mean(times),
                "std_final_time": math.sqrt(sum((t - mean(times)) ** 2 for t in times) / len(times)),
                "min_final_time": min(times),
                "max_final_time": max(times),
                "mean_rms_wz_err": mean(wz),
                "mean_rms_yaw_err": mean(yaw),
                "mean_rms_srbm_pred_err": mean(pred_s),
                "mean_rms_vicm_pred_err": mean(pred_v),
                "mean_wbc_delta_fr_norm": mean([g.mean_wbc_delta_fr_norm for g in group]),
            }
        )
    return rows


def paired_summary(results: list[TrialResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lambdas = sorted({r.lambda_scale for r in results})
    for lambda_scale in lambdas:
        srbm = [r for r in results if r.lambda_scale == lambda_scale and r.controller == "srbm"]
        vicm = [r for r in results if r.lambda_scale == lambda_scale and r.controller == "vicm"]
        if not srbm or not vicm:
            continue
        rows.append(
            {
                "lambda_scale": lambda_scale,
                "srbm_n": len(srbm),
                "vicm_n": len(vicm),
                "srbm_fall_count": sum(r.fall for r in srbm),
                "vicm_fall_count": sum(r.fall for r in vicm),
                "srbm_mean_final_time": mean([r.final_time for r in srbm]),
                "vicm_mean_final_time": mean([r.final_time for r in vicm]),
                "vicm_minus_srbm_final_time": mean([r.final_time for r in vicm])
                - mean([r.final_time for r in srbm]),
                "srbm_wz_minus_vicm_wz": mean([r.rms_wz_err for r in srbm])
                - mean([r.rms_wz_err for r in vicm]),
                "srbm_yaw_minus_vicm_yaw": mean([r.rms_yaw_err for r in srbm])
                - mean([r.rms_yaw_err for r in vicm]),
            }
        )
    return rows


def run_trial(
    out_dir: Path,
    lambda_scale: float,
    controller: str,
    rep: int,
    args: argparse.Namespace,
) -> TrialResult:
    case = f"lam{token(lambda_scale)}_{controller}_turn_posrot{token(args.posrot_att_scale)}_filtertau_r{rep}"
    env = os.environ.copy()
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_EXP": "1",
            "ODC_RUN_LABEL": case,
            "ODC_USE_LEG_LAMBDA_SCALE": "1",
            "ODC_LEG_LAMBDA_SCALE": f"{lambda_scale:.12g}",
            "ODC_TARGET_SPEED_X": f"{args.vx:.12g}",
            "ODC_TARGET_SPEED_Y": "0",
            "ODC_TSWING": f"{args.tswing:.12g}",
            "ODC_GAIT_SWITCH_FORCE_SOURCE": "touch",
            "ODC_SIM_END_TIME": f"{args.sim_end:.12g}",
            "ODC_TAU_BIAS_SCALE": f"{args.tau_bias_scale:.12g}",
            "ODC_TAU_NON_NORM_LIMIT": f"{args.tau_non_norm_limit:.12g}",
            "ODC_IG_DOT_FILTER_TAU": f"{args.ig_dot_filter_tau:.12g}",
            "ODC_PREDICT_IG_LINEAR": "0",
            "ODC_LINEAR_TAU_DYNAMICS": "1" if controller == "vicm" else "0",
            "ODC_MPC_L_DIAG": args.mpc_l_diag,
            "ODC_TORQUE_LIMIT_SCALE": f"{args.torque_limit_scale:.12g}",
            "ODC_WALK_LEG_PD_SCALE": f"{args.walk_leg_pd_scale:.12g}",
            "ODC_WBC_POSROT_POS_KP_SCALE": f"{args.posrot_pos_scale:.12g}",
            "ODC_WBC_POSROT_POS_KD_SCALE": f"{args.posrot_pos_scale:.12g}",
            "ODC_WBC_POSROT_ATT_KP_SCALE": f"{args.posrot_att_scale:.12g}",
            "ODC_WBC_POSROT_ATT_KD_SCALE": f"{args.posrot_att_scale:.12g}",
            "ODC_LOG_PREDICTION_ERROR": "1",
            "ODC_PRINT_MPC_TIMING": "0",
            "ODC_PRINT_FR_FF": "0",
            "ODC_SINE_TURN": "1",
            "ODC_SINE_WZ_BASE": "0",
            "ODC_SINE_WZ_AMP": f"{args.sine_wz_amp:.12g}",
            "ODC_SINE_WZ_PERIOD": f"{args.sine_wz_period:.12g}",
            "ODC_SINE_WZ_START_TIME": f"{args.sine_wz_start:.12g}",
        }
    )
    if controller == "vicm":
        env["ODC_USE_VICM"] = "1"
        env["ODC_USE_TAU_BIAS"] = "1"
    else:
        env["ODC_USE_VICM"] = "0"
        env["ODC_USE_TAU_BIAS"] = "0"

    log_path = out_dir / f"{case}.log"
    start = time.monotonic()
    with log_path.open("w") as log:
        subprocess.run(
            [str(BIN)],
            cwd=BUILD_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    wall_time_s = time.monotonic() - start

    trace_src = RECORD_DIR / "exp1_trace.csv"
    trace_dst = out_dir / f"{case}_trace.csv"
    shutil.copyfile(trace_src, trace_dst)

    datalog_src = RECORD_DIR / "exp1_datalog.log"
    if datalog_src.exists():
        shutil.copyfile(datalog_src, out_dir / f"{case}_datalog.log")

    pred_src = RECORD_DIR / f"pred_error_exp1_{case}_lf0.500000.csv"
    pred_dst = out_dir / f"{case}_pred_error.csv"
    if pred_src.exists():
        shutil.copyfile(pred_src, pred_dst)

    return parse_trial(
        case,
        controller,
        lambda_scale,
        rep,
        trace_dst,
        pred_dst,
        log_path,
        args.sim_end,
        wall_time_s,
        out_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--max-lambda", type=float, default=2.4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--posrot-pos-scale", type=float, default=1.0)
    parser.add_argument("--tau-bias-scale", type=float, default=1.0)
    parser.add_argument("--tau-non-norm-limit", type=float, default=0.0)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--sine-wz-amp", type=float, default=0.25)
    parser.add_argument("--sine-wz-period", type=float, default=4.0)
    parser.add_argument("--sine-wz-start", type=float, default=4.0)
    parser.add_argument("--torque-limit-scale", type=float, default=1.2)
    parser.add_argument("--walk-leg-pd-scale", type=float, default=1.2)
    parser.add_argument("--stop-threshold", type=float, default=8.0)
    parser.add_argument("--stop-consecutive", type=int, default=2)
    parser.add_argument(
        "--mpc-l-diag",
        default="50 50 80 1 200 1 1 1 10 100 10 1",
    )
    args = parser.parse_args()

    if not BIN.exists():
        raise FileNotFoundError(f"missing executable: {BIN}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / f"lambda_filter_turn_exp1_{stamp}"
    out_dir.mkdir(parents=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "lambda_filtered_vicm_first_then_srbm",
        "start": args.start,
        "step": args.step,
        "max_lambda": args.max_lambda,
        "repeats": args.repeats,
        "sim_end": args.sim_end,
        "vx": args.vx,
        "tswing": args.tswing,
        "posrot_att_scale": args.posrot_att_scale,
        "posrot_pos_scale": args.posrot_pos_scale,
        "tau_bias_scale": args.tau_bias_scale,
        "tau_non_norm_limit": args.tau_non_norm_limit,
        "ig_dot_filter_tau": args.ig_dot_filter_tau,
        "wbc_delta_fr_weight": "default_1e1",
        "wbc_delta_ddq_weight": "default_1e7",
        "sine_wz_amp": args.sine_wz_amp,
        "sine_wz_period": args.sine_wz_period,
        "sine_wz_start": args.sine_wz_start,
        "mpc_l_diag": args.mpc_l_diag,
    }
    write_csv(out_dir / "metadata.csv", [metadata])

    result_fields = list(TrialResult.__dataclass_fields__.keys())
    all_results: list[TrialResult] = []
    completed_lambdas: list[float] = []
    bad_streak = 0

    results_path = out_dir / "trials.csv"
    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()

        lambda_scale = args.start
        while lambda_scale <= args.max_lambda + 1e-9:
            lambda_scale = round(lambda_scale, 10)
            print(f"=== VICM lambda={lambda_scale:.3f} ===", flush=True)
            current: list[TrialResult] = []
            for rep in range(1, args.repeats + 1):
                result = run_trial(out_dir, lambda_scale, "vicm", rep, args)
                writer.writerow(result.__dict__)
                f.flush()
                all_results.append(result)
                current.append(result)
                print(
                    f"{result.case}: final={result.final_time:.3f}s "
                    f"fall={result.fall} wz_rms={result.rms_wz_err:.3f} "
                    f"yaw_rms={result.rms_yaw_err:.3f}",
                    flush=True,
                )

            completed_lambdas.append(lambda_scale)
            write_csv(out_dir / "summary.csv", summarize(all_results))
            unusable = (
                all(r.fall for r in current)
                and max(r.final_time for r in current) < args.stop_threshold
            )
            bad_streak = bad_streak + 1 if unusable else 0
            if lambda_scale >= 2.0 and bad_streak >= args.stop_consecutive:
                print(
                    f"Stop VICM: bad_streak={bad_streak} at lambda={lambda_scale:.3f}",
                    flush=True,
                )
                break
            lambda_scale = round(lambda_scale + args.step, 10)

        print("=== SRBM over completed lambda values ===", flush=True)
        for lambda_scale in completed_lambdas:
            print(f"=== SRBM lambda={lambda_scale:.3f} ===", flush=True)
            for rep in range(1, args.repeats + 1):
                result = run_trial(out_dir, lambda_scale, "srbm", rep, args)
                writer.writerow(result.__dict__)
                f.flush()
                all_results.append(result)
                print(
                    f"{result.case}: final={result.final_time:.3f}s "
                    f"fall={result.fall} wz_rms={result.rms_wz_err:.3f} "
                    f"yaw_rms={result.rms_yaw_err:.3f}",
                    flush=True,
                )
            write_csv(out_dir / "summary.csv", summarize(all_results))

    write_csv(out_dir / "paired_summary.csv", paired_summary(all_results))
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
