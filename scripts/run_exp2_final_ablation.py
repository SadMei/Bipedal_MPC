#!/usr/bin/env python3
"""Run the final four-model Experiment 2 comparison."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

from vicm_experiment_lib import (
    MPC_L_DIAG_MAIN,
    RECORD_DIR,
    append_csv,
    compute_trace_metrics,
    make_env,
    run_trial,
    token,
    write_csv,
)


CONTROLLERS = (
    ("srbm", "SRBM"),
    ("vicm_ig", "VI-CMPC"),
    ("ir_cmpc", "IR-CMPC"),
    ("ir_cmpc_hrel", "IRM-CMPC"),
)

TRIAL_FIELDS = [
    "case",
    "controller",
    "controller_label",
    "rep",
    "final_time",
    "completed_30s",
    "h10_wz_prediction_rmse",
    "wz_tracking_rmse",
    "velocity_tracking_rmse",
    "mpc_samples",
    "mpc_avg_wall_ms",
    "mpc_max_wall_ms",
    "mpc_avg_qp_ms",
    "mpc_max_qp_ms",
    "trace_path",
    "horizon_path",
    "log_path",
]

TIMING_RE = re.compile(
    r"\[MPC timing total\]\s+samples=(?P<samples>\d+)"
    r"\s+avg_wall_ms=(?P<avg_wall>[0-9.eE+-]+)"
    r"\s+max_wall_ms=(?P<max_wall>[0-9.eE+-]+)"
    r"\s+avg_qp_ms=(?P<avg_qp>[0-9.eE+-]+)"
    r"\s+max_qp_ms=(?P<max_qp>[0-9.eE+-]+)"
)


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def h10_wz_rmse(path: Path, start_time: float) -> float:
    errors: list[float] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                if int(row["horizon_steps"]) != 10:
                    continue
                if float(row["origin_time"]) < start_time:
                    continue
                error = float(row["err_wz"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(error):
                errors.append(error)
    if not errors:
        raise RuntimeError(f"No ten-step yaw-rate errors in {path}")
    return math.sqrt(statistics.mean(error * error for error in errors))


def timing_metrics(path: Path) -> dict[str, float | int]:
    matches = list(TIMING_RE.finditer(path.read_text(errors="replace")))
    if not matches:
        raise RuntimeError(f"No total MPC timing record in {path}")
    values = matches[-1].groupdict()
    return {
        "mpc_samples": int(values["samples"]),
        "mpc_avg_wall_ms": float(values["avg_wall"]),
        "mpc_max_wall_ms": float(values["max_wall"]),
        "mpc_avg_qp_ms": float(values["avg_qp"]),
        "mpc_max_qp_ms": float(values["max_qp"]),
    }


def summarize(
    rows: list[dict[str, object]],
    sim_end: float,
    controllers: tuple[tuple[str, str], ...] = CONTROLLERS,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for controller, label in controllers:
        group = [row for row in rows if row["controller"] == controller]
        survival = [float(row["final_time"]) for row in group]
        h10 = [float(row["h10_wz_prediction_rmse"]) for row in group]
        wz_tracking = [float(row["wz_tracking_rmse"]) for row in group]
        velocity_tracking = [float(row["velocity_tracking_rmse"]) for row in group]
        avg_wall = [float(row["mpc_avg_wall_ms"]) for row in group]
        max_wall = [float(row["mpc_max_wall_ms"]) for row in group]
        summary.append(
            {
                "controller": controller,
                "controller_label": label,
                "n": len(group),
                "completed_30s": sum(value >= sim_end - 1.0e-3 for value in survival),
                "mean_survival_time": statistics.mean(survival),
                "sample_sd_survival_time": sample_sd(survival),
                "mean_h10_wz_prediction_rmse": statistics.mean(h10),
                "sample_sd_h10_wz_prediction_rmse": sample_sd(h10),
                "mean_wz_tracking_rmse": statistics.mean(wz_tracking),
                "sample_sd_wz_tracking_rmse": sample_sd(wz_tracking),
                "mean_velocity_tracking_rmse": statistics.mean(velocity_tracking),
                "sample_sd_velocity_tracking_rmse": sample_sd(velocity_tracking),
                "mean_mpc_wall_ms": statistics.mean(avg_wall),
                "sample_sd_mpc_wall_ms": sample_sd(avg_wall),
                "max_mpc_wall_ms": max(max_wall),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-scale", type=float, default=1.8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--wz-amp", type=float, default=0.4)
    parser.add_argument("--wz-period", type=float, default=4.0)
    parser.add_argument("--wz-start", type=float, default=4.0)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--hrel-dot-filter-tau", type=float, default=0.02)
    parser.add_argument(
        "--hrel-contact-reset", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--controllers",
        default=",".join(controller for controller, _ in CONTROLLERS),
        help="Comma-separated controller variants to run.",
    )
    args = parser.parse_args()

    requested = [value.strip() for value in args.controllers.split(",") if value.strip()]
    labels = dict(CONTROLLERS)
    unknown = [value for value in requested if value not in labels]
    if unknown:
        raise ValueError(f"Unknown controller variants: {', '.join(unknown)}")
    selected_controllers = tuple((value, labels[value]) for value in requested)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / f"exp2_final_ablation_lam{token(args.lambda_scale)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "metadata.csv",
        [
            {
                **vars(args),
                "controllers": ",".join(
                    controller for controller, _ in selected_controllers
                ),
                "mpc_l_diag": MPC_L_DIAG_MAIN,
                "uncertainty": "none",
                "wbc_qp_limit": "1ms_wall_clock_and_nWSR_200",
                "qp_failure_fallback": "hold_previous_solution_without_decay",
                "h10_window": f"origin_time >= {args.wz_start} s",
            }
        ],
    )

    rows: list[dict[str, object]] = []
    trials_path = out_dir / "trials.csv"
    for rep in range(1, args.repeats + 1):
        for controller, label in selected_controllers:
            case = (
                f"exp2_lam{token(args.lambda_scale)}_amp{token(args.wz_amp)}_"
                f"{controller}_r{rep}"
            )
            env, pred_lf = make_env(
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
                mpc_l_diag=MPC_L_DIAG_MAIN,
                torque_limit_scale=1.2,
                walk_leg_pd_scale=1.2,
                hrel_dot_filter_tau=args.hrel_dot_filter_tau,
                hrel_reset_on_contact_switch=args.hrel_contact_reset,
                lambda_scale=args.lambda_scale,
                sine_turn=True,
                sine_wz_amp=args.wz_amp,
                sine_wz_period=args.wz_period,
                sine_wz_start=args.wz_start,
                gait_switch_threshold=100.0,
            )
            env["ODC_PRINT_MPC_TIMING"] = "1"
            env["ODC_MPC_TIMING_PRINT_INTERVAL"] = "1000"
            paths = run_trial(
                out_dir=out_dir,
                exp_id=2,
                case=case,
                env=env,
                pred_leg_fraction=pred_lf,
            )
            if paths.mpc_horizon is None:
                raise RuntimeError(f"Missing MPC horizon log for {case}")
            metrics = compute_trace_metrics(
                paths.trace,
                paths.pred,
                sim_end=args.sim_end,
                sine_start=args.wz_start,
                sine_period=args.wz_period,
                max_yaw_rms=999.0,
                max_wz_rms=999.0,
                max_path_rms=999.0,
                max_torso=999.0,
            )
            row: dict[str, object] = {
                "case": case,
                "controller": controller,
                "controller_label": label,
                "rep": rep,
                "final_time": metrics["final_time"],
                "completed_30s": int(float(metrics["final_time"]) >= args.sim_end - 1.0e-3),
                "h10_wz_prediction_rmse": h10_wz_rmse(
                    paths.mpc_horizon, args.wz_start
                ),
                "wz_tracking_rmse": metrics["rms_wz_err"],
                "velocity_tracking_rmse": metrics["rms_vel_err"],
                **timing_metrics(paths.log),
                "trace_path": paths.trace.name,
                "horizon_path": paths.mpc_horizon.name,
                "log_path": paths.log.name,
            }
            append_csv(trials_path, row, TRIAL_FIELDS)
            rows.append(row)
            print(
                f"{case}: survival={float(row['final_time']):.3f} s, "
                f"H10={float(row['h10_wz_prediction_rmse']):.3f} rad/s, "
                f"wz_track={float(row['wz_tracking_rmse']):.3f} rad/s, "
                f"MPC={float(row['mpc_avg_wall_ms']):.3f}/"
                f"{float(row['mpc_max_wall_ms']):.3f} ms",
                flush=True,
            )

    summary = summarize(rows, args.sim_end, selected_controllers)
    write_csv(out_dir / "summary.csv", summary)
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
