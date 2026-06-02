#!/usr/bin/env python3
"""Sequential leg-mass-fraction turning sweep for SRBM/VICM."""

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
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    i = 0
    while True:
        value = round(start + i * step, 10)
        if value > stop + 1e-9:
            break
        values.append(round(value, 6))
        i += 1
    return values


def rms(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(v * v for v in values) / len(values))


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def stdev_pop(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(mean([(v - avg) * (v - avg) for v in values]))


@dataclass
class TrialResult:
    case: str
    controller: str
    leg_mass_fraction: float
    posrot_att_scale: float
    rep: int
    fall: int
    fall_time: float
    final_time: float
    steps: int
    rms_wz_err: float
    max_abs_wz_err: float
    rms_yaw_err: float
    max_abs_yaw_err: float
    max_torso_angle_error: float
    rms_srbm_pred_err: float
    rms_vicm_pred_err: float
    controller_mass: float
    controller_leg_mass: float
    wall_time_s: float
    log_path: str
    trace_path: str
    pred_path: str


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def parse_trial(
    case: str,
    controller: str,
    leg_mass_fraction: float,
    posrot_att_scale: float,
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
        raise RuntimeError(f"empty trace for {case}")

    final_time = parse_float(rows[-1], "time")
    fall = 0
    fall_time = sim_end
    for row in rows:
        if parse_float(row, "fall_detected", 0.0) > 0.5:
            fall = 1
            fall_time = parse_float(row, "time")
            break
    if not fall:
        fall_time = final_time
    steps = int(parse_float(rows[-1], "step_count", 0.0))
    controller_mass = parse_float(rows[-1], "controller_mass")
    controller_leg_mass = parse_float(rows[-1], "controller_leg_mass")

    eval_rows = [r for r in rows if parse_float(r, "time") >= 4.0]
    if not eval_rows:
        eval_rows = rows
    wz_err = [parse_float(r, "wz") - parse_float(r, "wz_ref") for r in eval_rows]
    yaw_err = [parse_float(r, "yaw") - parse_float(r, "yaw_ref") for r in eval_rows]
    torso = [parse_float(r, "torso_angle_error") for r in eval_rows]

    srbm_pred_err: list[float] = []
    vicm_pred_err: list[float] = []
    if pred_path.exists():
        with pred_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if parse_float(row, "time") >= 4.0:
                    srbm_pred_err.append(parse_float(row, "srbm_err_norm"))
                    vicm_pred_err.append(parse_float(row, "vicm_err_norm"))

    rel_log = str(log_path.relative_to(out_dir))
    rel_trace = str(trace_path.relative_to(out_dir))
    rel_pred = str(pred_path.relative_to(out_dir)) if pred_path.exists() else ""
    return TrialResult(
        case=case,
        controller=controller,
        leg_mass_fraction=leg_mass_fraction,
        posrot_att_scale=posrot_att_scale,
        rep=rep,
        fall=fall,
        fall_time=fall_time,
        final_time=final_time,
        steps=steps,
        rms_wz_err=rms(wz_err),
        max_abs_wz_err=max((abs(v) for v in wz_err), default=math.nan),
        rms_yaw_err=rms(yaw_err),
        max_abs_yaw_err=max((abs(v) for v in yaw_err), default=math.nan),
        max_torso_angle_error=max(torso, default=math.nan),
        rms_srbm_pred_err=rms(srbm_pred_err),
        rms_vicm_pred_err=rms(vicm_pred_err),
        controller_mass=controller_mass,
        controller_leg_mass=controller_leg_mass,
        wall_time_s=wall_time_s,
        log_path=rel_log,
        trace_path=rel_trace,
        pred_path=rel_pred,
    )


def summarize(results: list[TrialResult]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str], list[TrialResult]] = {}
    for result in results:
        groups.setdefault((result.leg_mass_fraction, result.controller), []).append(result)

    rows: list[dict[str, object]] = []
    for (leg_mass_fraction, controller), group in sorted(groups.items()):
        final_times = [g.final_time for g in group]
        rows.append(
            {
                "leg_mass_fraction": leg_mass_fraction,
                "controller": controller,
                "n": len(group),
                "fall_count": sum(g.fall for g in group),
                "mean_final_time": mean(final_times),
                "std_final_time": stdev_pop(final_times),
                "min_final_time": min(final_times),
                "max_final_time": max(final_times),
                "mean_rms_wz_err": mean([g.rms_wz_err for g in group]),
                "mean_rms_yaw_err": mean([g.rms_yaw_err for g in group]),
                "mean_max_torso_angle_error": mean([g.max_torso_angle_error for g in group]),
                "mean_rms_srbm_pred_err": mean([g.rms_srbm_pred_err for g in group]),
                "mean_rms_vicm_pred_err": mean([g.rms_vicm_pred_err for g in group]),
                "mean_controller_mass": mean([g.controller_mass for g in group]),
                "mean_controller_leg_mass": mean([g.controller_leg_mass for g in group]),
            }
        )
    return rows


def paired_summary(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_lf: dict[float, dict[str, dict[str, object]]] = {}
    for row in summary_rows:
        by_lf.setdefault(float(row["leg_mass_fraction"]), {})[str(row["controller"])] = row

    rows: list[dict[str, object]] = []
    for leg_mass_fraction, controllers in sorted(by_lf.items()):
        if "srbm" not in controllers or "vicm" not in controllers:
            continue
        srbm = controllers["srbm"]
        vicm = controllers["vicm"]
        rows.append(
            {
                "leg_mass_fraction": leg_mass_fraction,
                "vicm_minus_srbm_final_time": float(vicm["mean_final_time"])
                - float(srbm["mean_final_time"]),
                "srbm_fall_count": srbm["fall_count"],
                "vicm_fall_count": vicm["fall_count"],
                "srbm_wz_minus_vicm_wz": float(srbm["mean_rms_wz_err"])
                - float(vicm["mean_rms_wz_err"]),
                "srbm_yaw_minus_vicm_yaw": float(srbm["mean_rms_yaw_err"])
                - float(vicm["mean_rms_yaw_err"]),
                "srbm_pred_minus_vicm_pred_on_srbm_runs": float(
                    srbm["mean_rms_srbm_pred_err"]
                )
                - float(srbm["mean_rms_vicm_pred_err"]),
                "srbm_pred_minus_vicm_pred_on_vicm_runs": float(
                    vicm["mean_rms_srbm_pred_err"]
                )
                - float(vicm["mean_rms_vicm_pred_err"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    if fields is None:
        fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_trial(
    out_dir: Path,
    leg_mass_fraction: float,
    controller: str,
    rep: int,
    args: argparse.Namespace,
) -> TrialResult:
    case = (
        f"lf{token(leg_mass_fraction)}_{controller}_turn_"
        f"posrot{token(args.posrot_att_scale)}_r{rep}"
    )
    env = os.environ.copy()
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_EXP": "1",
            "ODC_RUN_LABEL": case,
            "ODC_USE_LEG_LAMBDA_SCALE": "0",
            "ODC_LEG_MASS_FRACTION": f"{leg_mass_fraction:.12g}",
            "ODC_TARGET_SPEED_X": f"{args.vx:.12g}",
            "ODC_TARGET_SPEED_Y": "0",
            "ODC_TSWING": f"{args.tswing:.12g}",
            "ODC_GAIT_SWITCH_FORCE_SOURCE": "touch",
            "ODC_SIM_END_TIME": f"{args.sim_end:.12g}",
            "ODC_TAU_BIAS_SCALE": f"{args.tau_bias_scale:.12g}",
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
    if args.wbc_delta_fr_weight is not None:
        env["ODC_WBC_DELTA_FR_WEIGHT"] = f"{args.wbc_delta_fr_weight:.12g}"
    if args.wbc_delta_ddq_weight is not None:
        env["ODC_WBC_DELTA_DDQ_WEIGHT"] = f"{args.wbc_delta_ddq_weight:.12g}"

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

    pred_src = (
        RECORD_DIR
        / f"pred_error_exp1_{case}_lf{leg_mass_fraction:.6f}.csv"
    )
    pred_dst = out_dir / f"{case}_pred_error.csv"
    if pred_src.exists():
        shutil.copyfile(pred_src, pred_dst)

    return parse_trial(
        case,
        controller,
        leg_mass_fraction,
        args.posrot_att_scale,
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
    parser.add_argument("--start", type=float, default=0.60)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--controllers",
        default="srbm,vicm",
        help="Comma-separated controllers to run: srbm,vicm",
    )
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--posrot-pos-scale", type=float, default=1.0)
    parser.add_argument("--tau-bias-scale", type=float, default=0.5)
    parser.add_argument("--sine-wz-amp", type=float, default=0.25)
    parser.add_argument("--sine-wz-period", type=float, default=4.0)
    parser.add_argument("--sine-wz-start", type=float, default=4.0)
    parser.add_argument("--torque-limit-scale", type=float, default=1.2)
    parser.add_argument("--walk-leg-pd-scale", type=float, default=1.2)
    parser.add_argument("--wbc-delta-fr-weight", type=float, default=None)
    parser.add_argument("--wbc-delta-ddq-weight", type=float, default=None)
    parser.add_argument(
        "--mpc-l-diag",
        default="50 50 20 1 200 1 1 1 2 100 10 1",
    )
    args = parser.parse_args()

    if not BIN.exists():
        raise FileNotFoundError(f"missing executable: {BIN}")
    controllers = [c.strip().lower() for c in args.controllers.split(",") if c.strip()]
    unknown = [c for c in controllers if c not in ("srbm", "vicm")]
    if unknown:
        raise ValueError(f"unknown controller(s): {', '.join(unknown)}")
    if not controllers:
        raise ValueError("at least one controller is required")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / (
        f"posrot_att{token(args.posrot_att_scale)}_"
        f"legfrac{token(args.start)}_{token(args.stop)}_"
        f"turn_rep{args.repeats}_{stamp}"
    )
    out_dir.mkdir(parents=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "start": args.start,
        "stop": args.stop,
        "step": args.step,
        "repeats": args.repeats,
        "sim_end": args.sim_end,
        "vx": args.vx,
        "tswing": args.tswing,
        "posrot_att_scale": args.posrot_att_scale,
        "posrot_pos_scale": args.posrot_pos_scale,
        "tau_bias_scale": args.tau_bias_scale,
        "sine_wz_amp": args.sine_wz_amp,
        "sine_wz_period": args.sine_wz_period,
        "sine_wz_start": args.sine_wz_start,
        "mpc_l_diag": args.mpc_l_diag,
        "wbc_delta_fr_weight": args.wbc_delta_fr_weight,
        "wbc_delta_ddq_weight": args.wbc_delta_ddq_weight,
    }
    write_csv(out_dir / "metadata.csv", [metadata])

    result_fields = list(TrialResult.__dataclass_fields__.keys())
    all_results: list[TrialResult] = []
    results_path = out_dir / "trials.csv"
    summary_path = out_dir / "summary.csv"
    paired_path = out_dir / "paired_summary.csv"

    with results_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()
        for leg_mass_fraction in frange(args.start, args.stop, args.step):
            print(f"=== rho_l={leg_mass_fraction:.3f} ===", flush=True)
            lf_results: list[TrialResult] = []
            for controller in controllers:
                for rep in range(1, args.repeats + 1):
                    result = run_trial(out_dir, leg_mass_fraction, controller, rep, args)
                    writer.writerow(result.__dict__)
                    f.flush()
                    all_results.append(result)
                    lf_results.append(result)
                    print(
                        f"{result.case}: final={result.final_time:.3f}s "
                        f"fall={result.fall} wz_rms={result.rms_wz_err:.3f} "
                        f"yaw_rms={result.rms_yaw_err:.3f} "
                        f"pred_delta={result.rms_srbm_pred_err - result.rms_vicm_pred_err:.3f}",
                        flush=True,
                    )

            summary_rows = summarize(all_results)
            write_csv(summary_path, summary_rows)
            pair_rows = paired_summary(summary_rows)
            write_csv(paired_path, pair_rows)
            current_pair = [p for p in pair_rows if float(p["leg_mass_fraction"]) == leg_mass_fraction]
            if current_pair:
                p = current_pair[0]
                print(
                    f"rho_l={leg_mass_fraction:.3f} paired: "
                    f"dt={float(p['vicm_minus_srbm_final_time']):+.3f}s "
                    f"wz_delta={float(p['srbm_wz_minus_vicm_wz']):+.3f} "
                    f"yaw_delta={float(p['srbm_yaw_minus_vicm_yaw']):+.3f}",
                    flush=True,
                )

    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
