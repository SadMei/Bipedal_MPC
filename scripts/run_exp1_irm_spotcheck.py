#!/usr/bin/env python3
"""Run deterministic IRM-CMPC spot checks under Experiment 1 settings."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from datetime import datetime
from pathlib import Path

from vicm_experiment_lib import (
    MPC_L_DIAG_MAIN,
    RECORD_DIR,
    compute_trace_metrics,
    make_env,
    paired_uncertainty_profile,
    run_trial,
    token,
    write_csv,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambda-values", default="1.0 1.2 1.4 1.6 2.2"
    )
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
    parser.add_argument("--uncertainty-rep", type=int, default=1)
    parser.add_argument("--uncertainty-base-seed", type=int, default=2026072200)
    args = parser.parse_args()

    lambdas = [
        float(value) for value in args.lambda_values.replace(",", " ").split()
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / f"exp1_irm_spotcheck_{stamp}"
    out_dir.mkdir(parents=True)
    uncertainty = paired_uncertainty_profile(
        args.uncertainty_rep, args.uncertainty_base_seed
    )
    write_csv(
        out_dir / "metadata.csv",
        [
            {
                **vars(args),
                "controller": "ir_cmpc_hrel",
                "controller_label": "IRM-CMPC",
                "mpc_l_diag": MPC_L_DIAG_MAIN,
                **uncertainty,
                "wbc_qp_limit": "1ms_wall_clock_and_nWSR_200",
                "qp_failure_fallback": "hold_previous_solution_without_decay",
            }
        ],
    )

    rows: list[dict[str, object]] = []
    for lambda_scale in lambdas:
        case = f"exp1_irm_spot_lam{token(lambda_scale)}"
        env, pred_lf = make_env(
            exp_id=1,
            case=case,
            controller="ir_cmpc_hrel",
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
            lambda_scale=lambda_scale,
            sine_turn=True,
            sine_wz_amp=args.wz_amp,
            sine_wz_period=args.wz_period,
            sine_wz_start=args.wz_start,
            push_force=float(uncertainty["push_force"]),
            push_start=float(uncertainty["push_start"]),
            push_duration=float(uncertainty["push_duration"]),
            push_dir_x=float(uncertainty["push_dir_x"]),
            push_dir_y=float(uncertainty["push_dir_y"]),
            gait_switch_threshold=100.0,
            sensor_noise_enable=bool(uncertainty["noise_enabled"]),
            sensor_noise_seed=int(uncertainty["noise_seed"]),
            noise_base_pos_std=float(uncertainty["noise_base_pos_std"]),
            noise_base_rpy_std=float(uncertainty["noise_base_rpy_std"]),
            noise_base_vel_std=float(uncertainty["noise_base_vel_std"]),
            noise_base_omega_std=float(uncertainty["noise_base_omega_std"]),
            noise_joint_pos_std=float(uncertainty["noise_joint_pos_std"]),
            noise_joint_vel_std=float(uncertainty["noise_joint_vel_std"]),
            noise_foot_force_std=float(uncertainty["noise_foot_force_std"]),
        )
        paths = run_trial(
            out_dir=out_dir,
            exp_id=1,
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
            "lambda_scale": lambda_scale,
            "controller": "ir_cmpc_hrel",
            "controller_label": "IRM-CMPC",
            "final_time": metrics["final_time"],
            "completed_30s": int(
                float(metrics["final_time"]) >= args.sim_end - 1.0e-3
            ),
            "h10_wz_prediction_rmse": h10_wz_rmse(
                paths.mpc_horizon, args.wz_start
            ),
            "wz_tracking_rmse": metrics["rms_wz_err"],
            "velocity_tracking_rmse": metrics["rms_vel_err"],
            "trace_path": paths.trace.name,
            "horizon_path": paths.mpc_horizon.name,
            "log_path": paths.log.name,
        }
        rows.append(row)
        write_csv(out_dir / "summary.csv", rows)
        print(
            f"lambda={lambda_scale:.1f}: survival={float(row['final_time']):.3f} s, "
            f"H10={float(row['h10_wz_prediction_rmse']):.4f} rad/s, "
            f"wz_track={float(row['wz_tracking_rmse']):.4f} rad/s, "
            f"vel_track={float(row['velocity_tracking_rmse']):.4f} m/s",
            flush=True,
        )

    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
