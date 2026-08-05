#!/usr/bin/env python3
"""Experiment 2: discrete-momentum model ablation at fixed leg inertia."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

from vicm_experiment_lib import (
    MPC_L_DIAG_MAIN,
    RECORD_DIR,
    append_csv,
    make_env,
    run_trial,
    token,
    write_csv,
)


CONTROLLERS = ("srbm", "vicm_ig", "dm_frozen", "dm_preview")
LABELS = {
    "srbm": "SRBM",
    "vicm_ig": "VI-CMPC",
    "dm_frozen": "DM-CMPC-FI",
    "dm_preview": "DM-CMPC",
}
TRIAL_FIELDS = (
    "case",
    "controller",
    "controller_label",
    "lambda_scale",
    "rep",
    "fall",
    "final_time",
    "rms_wz_err",
    "rms_tracking_err",
    "mpc_qp_fail_frames",
    "wbc_qp_fail_frames",
    "wall_time_s",
    "trace_path",
    "pred_path",
    "horizon_path",
    "log_path",
)


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def parse_metrics(trace_path: Path, eval_start: float, sim_end: float) -> dict[str, object]:
    with trace_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")

    final_time = float(rows[-1]["time"])
    fall = int(float(rows[-1]["fall_detected"]))
    evaluation = [row for row in rows if float(row["time"]) >= eval_start]
    wz_error = [
        float(row["wz"]) - float(row["wz_ref"]) for row in evaluation
    ]
    tracking_error = [float(row["vel_track_error"]) for row in evaluation]
    return {
        "fall": fall,
        "final_time": final_time if fall else sim_end,
        "rms_wz_err": rms(wz_error),
        "rms_tracking_err": rms(tracking_error),
        "mpc_qp_fail_frames": sum(
            int(float(row.get("mpc_qp_status", "0"))) != 0 for row in rows
        ),
        "wbc_qp_fail_frames": sum(
            int(float(row.get("wbc_qp_status", "0"))) != 0 for row in rows
        ),
    }


def summarize(
    rows: list[dict[str, object]], controllers: tuple[str, ...]
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for controller in controllers:
        group = [row for row in rows if row["controller"] == controller]
        result: dict[str, object] = {
            "controller": controller,
            "controller_label": LABELS[controller],
            "n": len(group),
            "fall_count": sum(int(row["fall"]) for row in group),
            "completed_count": sum(not int(row["fall"]) for row in group),
        }
        for field in ("final_time", "rms_wz_err", "rms_tracking_err"):
            values = [float(row[field]) for row in group]
            result[f"mean_{field}"] = statistics.mean(values)
            result[f"std_{field}"] = statistics.pstdev(values)
            result[f"min_{field}"] = min(values)
            result[f"max_{field}"] = max(values)
        result["total_mpc_qp_fail_frames"] = sum(
            int(row["mpc_qp_fail_frames"]) for row in group
        )
        result["total_wbc_qp_fail_frames"] = sum(
            int(row["wbc_qp_fail_frames"]) for row in group
        )
        summary.append(result)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-scale", type=float, default=1.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--wz-amp", type=float, default=0.4)
    parser.add_argument("--wz-period", type=float, default=4.0)
    parser.add_argument("--wz-start", type=float, default=4.0)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--mpc-l-diag", default=MPC_L_DIAG_MAIN)
    parser.add_argument(
        "--controllers",
        default=",".join(CONTROLLERS),
        help="comma-separated subset of srbm,vicm_ig,dm_frozen,dm_preview",
    )
    args = parser.parse_args()
    controllers = tuple(
        controller.strip() for controller in args.controllers.split(",")
        if controller.strip()
    )
    unknown = [controller for controller in controllers if controller not in CONTROLLERS]
    if not controllers or unknown:
        parser.error(f"invalid controller subset: {args.controllers}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        RECORD_DIR
        / f"exp2_discrete_momentum_ablation_lam{token(args.lambda_scale)}_{stamp}"
    )
    out_dir.mkdir(parents=True)
    write_csv(
        out_dir / "metadata.csv",
        [
            {
                **vars(args),
                "controllers": ",".join(controllers),
                "uncertainty": "none",
                "push_force": 0.0,
                "metric_window": f"t >= {args.wz_start} s through trial termination",
                "wbc_qp_limit": "1ms_wall_clock_and_nWSR_200",
                "qp_failure_fallback": "hold_previous_solution_without_decay",
            }
        ],
    )

    rows: list[dict[str, object]] = []
    trials_path = out_dir / "trials.csv"
    for controller in controllers:
        label = LABELS[controller]
        print(f"=== {label} ===", flush=True)
        for rep in range(1, args.repeats + 1):
            case = (
                f"exp2_lam{token(args.lambda_scale)}_amp{token(args.wz_amp)}_"
                f"{controller}_r{rep}"
            )
            env, prediction_fraction = make_env(
                exp_id=2,
                case=case,
                controller=controller,
                sim_end=args.sim_end,
                vx=args.vx,
                tswing=args.tswing,
                posrot_att_scale=args.posrot_att_scale,
                posrot_pos_scale=1.0,
                tau_bias_scale=1.0,
                tau_non_norm_limit=0.0,
                ig_dot_filter_tau=args.ig_dot_filter_tau,
                mpc_l_diag=args.mpc_l_diag,
                torque_limit_scale=1.2,
                walk_leg_pd_scale=1.2,
                lambda_scale=args.lambda_scale,
                sine_turn=True,
                sine_wz_amp=args.wz_amp,
                sine_wz_period=args.wz_period,
                sine_wz_start=args.wz_start,
                push_force=0.0,
                push_duration=0.0,
                sensor_noise_enable=False,
            )
            start = time.monotonic()
            paths = run_trial(
                out_dir=out_dir,
                exp_id=2,
                case=case,
                env=env,
                pred_leg_fraction=prediction_fraction,
            )
            metrics = parse_metrics(paths.trace, args.wz_start, args.sim_end)
            row = {
                "case": case,
                "controller": controller,
                "controller_label": label,
                "lambda_scale": args.lambda_scale,
                "rep": rep,
                **metrics,
                "wall_time_s": time.monotonic() - start,
                "trace_path": str(paths.trace.relative_to(out_dir)),
                "pred_path": (
                    str(paths.pred.relative_to(out_dir)) if paths.pred else ""
                ),
                "horizon_path": (
                    str(paths.mpc_horizon.relative_to(out_dir))
                    if paths.mpc_horizon
                    else ""
                ),
                "log_path": str(paths.log.relative_to(out_dir)),
            }
            append_csv(trials_path, row, list(TRIAL_FIELDS))
            rows.append(row)
            print(
                f"{case}: final={float(row['final_time']):.3f}s "
                f"fall={row['fall']} wz_rms={float(row['rms_wz_err']):.3f} "
                f"tracking_rms={float(row['rms_tracking_err']):.3f}",
                flush=True,
            )

    write_csv(out_dir / "summary.csv", summarize(rows, controllers))
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
