#!/usr/bin/env python3
"""Experiment 3: VICM component ablation around the informative boundary."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from vicm_experiment_lib import (
    CONTROLLER_LABELS,
    FIGURE_DIR,
    MPC_L_DIAG_MAIN,
    RECORD_DIR,
    aggregate,
    append_csv,
    compute_trace_metrics,
    make_env,
    mean,
    parse_controller_list,
    paired_uncertainty_profile,
    plot_tracking,
    run_trial,
    setup_matplotlib,
    stdev_pop,
    token,
    write_csv,
)


TRIAL_FIELDS = [
    "case",
    "controller",
    "controller_label",
    "lambda_scale",
    "wz_amp",
    "rep",
    "noise_enabled",
    "noise_seed",
    "push_force",
    "push_start",
    "push_duration",
    "push_angle_deg",
    "fall",
    "fall_time",
    "final_time",
    "steps",
    "mpc_qp_fail_frames",
    "wbc_qp_fail_frames",
    "task_valid",
    "rms_yaw_err",
    "rms_wz_err",
    "max_torso_angle_error",
    "rms_srbm_pred_err",
    "rms_vi_pred_err",
    "rms_ir_pred_err",
    "rms_ir_nf_pred_err",
    "rms_vicm_pred_err",
    "rms_tau_mpc_norm",
    "mean_wbc_delta_fr_norm",
    "wall_time_s",
    "trace_path",
    "pred_path",
    "log_path",
]


def plot_ablation(out_dir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    summary = aggregate(
        rows,
        ["controller_label"],
        [
            "final_time",
            "rms_yaw_err",
            "rms_wz_err",
            "max_torso_angle_error",
            "rms_srbm_pred_err",
            "rms_vi_pred_err",
            "rms_ir_pred_err",
            "rms_ir_nf_pred_err",
            "rms_vicm_pred_err",
            "rms_tau_mpc_norm",
            "mean_wbc_delta_fr_norm",
        ],
    )
    order = ["SRBM", "VICM-Ig", "VICM-Ac", "VICM-Ac no filter", "VICM affine tau"]
    summary.sort(key=lambda r: order.index(str(r["controller_label"])) if str(r["controller_label"]) in order else 99)
    write_csv(out_dir / "summary.csv", summary)
    prediction_models = [
        ("SRBM", "rms_srbm_pred_err"),
        ("VI-CMPC", "rms_vi_pred_err"),
        ("IR-CMPC", "rms_ir_pred_err"),
        ("IR-CMPC-NF", "rms_ir_nf_pred_err"),
    ]
    prediction_summary = []
    for model, field in prediction_models:
        values = [float(row[field]) for row in rows]
        prediction_summary.append(
            {
                "prediction_model": model,
                "n_trajectories": len(values),
                "mean_rms_error": mean(values),
                "std_rms_error": stdev_pop(values),
                "min_rms_error": min(values),
                "max_rms_error": max(values),
            }
        )
    write_csv(out_dir / "prediction_model_summary.csv", prediction_summary)

    labels = [str(r["controller_label"]) for r in summary]
    colors = ["#5477C4", "#71B436", "#CC6F47", "#BD569B", "#736422"][: len(labels)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    specs = [
        ("mean_final_time", "final time [s]", "Closed-loop survival"),
        ("mean_rms_ir_pred_err", "one-step omega error [rad/s]", "Inertia-rate prediction error"),
        ("mean_rms_wz_err", "wz RMS [rad/s]", "Yaw-rate response error"),
        ("mean_max_torso_angle_error", "torso max [rad]", "Torso disturbance"),
    ]
    for ax, (field, ylabel, title) in zip(axes.ravel(), specs):
        values = [float(r[field]) for r in summary]
        ax.bar(labels, values, color=colors, edgecolor="#464C55", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.6)
    axes[0, 0].text(
        0.0,
        1.12,
        "Experiment 3 model-component ablation",
        transform=axes[0, 0].transAxes,
        fontsize=13,
        color="#1F2430",
    )
    axes[0, 0].text(
        0.0,
        1.04,
        "SRBM, configuration-dependent inertia only, and Ac-injected Ig-dot dynamics under the same WBC/gait settings.",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        color="#6F768A",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "exp3_ablation_summary.png", dpi=180)
    fig.savefig(FIGURE_DIR / f"exp3_ablation_summary_{out_dir.name}.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-scale", type=float, default=1.7)
    parser.add_argument("--wz-amp", type=float, default=0.4)
    parser.add_argument("--wz-period", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--controllers", default="srbm,vicm_ig,vicm_ac")
    parser.add_argument("--include-filter-ablation", action="store_true")
    parser.add_argument("--include-affine-tau", action="store_true")
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--posrot-pos-scale", type=float, default=1.0)
    parser.add_argument("--tau-bias-scale", type=float, default=1.0)
    parser.add_argument("--tau-non-norm-limit", type=float, default=0.0)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--sine-wz-start", type=float, default=4.0)
    parser.add_argument("--torque-limit-scale", type=float, default=1.2)
    parser.add_argument("--walk-leg-pd-scale", type=float, default=1.2)
    parser.add_argument("--gait-switch-threshold", type=float, default=100.0)
    parser.add_argument("--paired-light-uncertainty", action="store_true")
    parser.add_argument("--uncertainty-base-seed", type=int, default=2026072200)
    parser.add_argument("--mpc-l-diag", default=MPC_L_DIAG_MAIN)
    parser.add_argument("--max-yaw-rms", type=float, default=0.12)
    parser.add_argument("--max-wz-rms", type=float, default=0.65)
    parser.add_argument("--max-path-rms", type=float, default=999.0)
    parser.add_argument("--max-torso", type=float, default=0.15)
    args = parser.parse_args()

    controllers = parse_controller_list(args.controllers)
    if args.include_filter_ablation and "vicm_ac_nofilter" not in controllers:
        controllers.append("vicm_ac_nofilter")
    if args.include_affine_tau and "vicm_affine" not in controllers:
        controllers.append("vicm_affine")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / f"exp3_model_ablation_lam{token(args.lambda_scale)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    metadata = {
        **vars(args),
        "paper_experiment": "Experiment 2: model ablation",
        "wbc_qp_limit": "1ms_wall_clock_and_nWSR_200",
        "mpc_qp_limit": "5ms_wall_clock",
        "qp_failure_fallback": "hold_previous_solution_without_decay",
    }
    write_csv(out_dir / "metadata.csv", [metadata])
    write_csv(
        out_dir / "uncertainty_profiles.csv",
        [
            {"rep": rep, **paired_uncertainty_profile(rep, args.uncertainty_base_seed)}
            for rep in range(1, args.repeats + 1)
        ],
    )

    rows: list[dict[str, object]] = []
    traces_for_plot: dict[str, Path] = {}
    trials_path = out_dir / "trials.csv"

    for controller in controllers:
        label = CONTROLLER_LABELS[controller]
        print(f"=== exp3 {label} ===", flush=True)
        for rep in range(1, args.repeats + 1):
            uncertainty = paired_uncertainty_profile(rep, args.uncertainty_base_seed)
            if not args.paired_light_uncertainty:
                uncertainty = paired_uncertainty_profile(1, args.uncertainty_base_seed)
            case = f"exp3_lam{token(args.lambda_scale)}_amp{token(args.wz_amp)}_{controller}_r{rep}"
            env, pred_lf = make_env(
                exp_id=3,
                case=case,
                controller=controller,
                sim_end=args.sim_end,
                vx=args.vx,
                tswing=args.tswing,
                posrot_att_scale=args.posrot_att_scale,
                posrot_pos_scale=args.posrot_pos_scale,
                tau_bias_scale=args.tau_bias_scale,
                tau_non_norm_limit=args.tau_non_norm_limit,
                ig_dot_filter_tau=args.ig_dot_filter_tau,
                mpc_l_diag=args.mpc_l_diag,
                torque_limit_scale=args.torque_limit_scale,
                walk_leg_pd_scale=args.walk_leg_pd_scale,
                lambda_scale=args.lambda_scale,
                sine_turn=True,
                sine_wz_amp=args.wz_amp,
                sine_wz_period=args.wz_period,
                sine_wz_start=args.sine_wz_start,
                gait_switch_threshold=args.gait_switch_threshold,
                push_force=float(uncertainty["push_force"]),
                push_start=float(uncertainty["push_start"]),
                push_duration=float(uncertainty["push_duration"]),
                push_dir_x=float(uncertainty["push_dir_x"]),
                push_dir_y=float(uncertainty["push_dir_y"]),
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
            start = time.monotonic()
            paths = run_trial(out_dir=out_dir, exp_id=3, case=case, env=env, pred_leg_fraction=pred_lf)
            wall_time = time.monotonic() - start
            metrics = compute_trace_metrics(
                paths.trace,
                paths.pred,
                sim_end=args.sim_end,
                sine_start=args.sine_wz_start,
                sine_period=args.wz_period,
                max_yaw_rms=args.max_yaw_rms,
                max_wz_rms=args.max_wz_rms,
                max_path_rms=args.max_path_rms,
                max_torso=args.max_torso,
            )
            row = {
                "case": case,
                "controller": controller,
                "controller_label": label,
                "lambda_scale": args.lambda_scale,
                "wz_amp": args.wz_amp,
                "rep": rep,
                "noise_enabled": int(bool(uncertainty["noise_enabled"])),
                "noise_seed": int(uncertainty["noise_seed"]),
                "push_force": float(uncertainty["push_force"]),
                "push_start": float(uncertainty["push_start"]),
                "push_duration": float(uncertainty["push_duration"]),
                "push_angle_deg": float(uncertainty["push_angle_deg"]),
                **metrics,
                "wall_time_s": wall_time,
                "trace_path": str(paths.trace.relative_to(out_dir)),
                "pred_path": str(paths.pred.relative_to(out_dir)) if paths.pred else "",
                "log_path": str(paths.log.relative_to(out_dir)),
            }
            append_csv(trials_path, row, TRIAL_FIELDS)
            rows.append(row)
            if rep == 1:
                traces_for_plot[label] = paths.trace
            print(
                f"{case}: final={float(metrics['final_time']):.3f}s "
                f"fall={metrics['fall']} valid={metrics['task_valid']} "
                f"pred_s={float(metrics['rms_srbm_pred_err']):.3f} "
                f"pred_vi={float(metrics['rms_vi_pred_err']):.3f} "
                f"pred_ir={float(metrics['rms_ir_pred_err']):.3f}",
                flush=True,
            )

    plot_ablation(out_dir, rows)
    if traces_for_plot:
        plot_tracking(
            traces_for_plot,
            out_dir / "exp3_ablation_tracking.png",
            title="Experiment 3 representative ablation tracking",
            subtitle=f"lambda={args.lambda_scale:.2f}, wz_amp={args.wz_amp:.2f}; curves are smoothed over 0.20 s",
            t_min=args.sine_wz_start,
            t_max=min(args.sim_end, args.sine_wz_start + 12.0),
        )
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
