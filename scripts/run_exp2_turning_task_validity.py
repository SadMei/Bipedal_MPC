#!/usr/bin/env python3
"""Experiment 2: task-valid sinusoidal turning range for SRBM vs VICM."""

from __future__ import annotations

import argparse
import csv
import math
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
    parse_float_list,
    plot_tracking,
    plot_xy_tracking,
    run_trial,
    setup_matplotlib,
    token,
    write_csv,
)


TRIAL_FIELDS = [
    "case",
    "controller",
    "controller_label",
    "lambda_scale",
    "wz_amp",
    "wz_period",
    "rep",
    "fall",
    "fall_time",
    "final_time",
    "steps",
    "task_valid",
    "eval_duration",
    "rms_yaw_err",
    "rms_wz_err",
    "rms_path_err",
    "max_torso_angle_error",
    "wz_gain",
    "wz_phase_lag_s",
    "yaw_gain",
    "yaw_phase_lag_s",
    "wz_high_freq_residual_rms",
    "rms_srbm_pred_err",
    "rms_vicm_pred_err",
    "mean_wbc_delta_fr_norm",
    "max_wbc_delta_fr_norm",
    "wall_time_s",
    "trace_path",
    "pred_path",
    "log_path",
]


def plot_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    summary = aggregate(
        rows,
        ["wz_amp", "controller_label"],
        [
            "final_time",
            "rms_yaw_err",
            "rms_wz_err",
            "max_torso_angle_error",
            "wz_phase_lag_s",
            "wz_high_freq_residual_rms",
        ],
    )
    write_csv(out_dir / "summary.csv", summary)

    colors = {"SRBM": "#5477C4", "VICM-Ac": "#CC6F47"}
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.8))
    for label in sorted({str(r["controller_label"]) for r in summary}):
        group = [r for r in summary if str(r["controller_label"]) == label]
        group.sort(key=lambda r: float(r["wz_amp"]))
        amps = [float(r["wz_amp"]) for r in group]
        ax.errorbar(
            amps,
            [float(r["mean_final_time"]) for r in group],
            yerr=[float(r["std_final_time"]) for r in group],
            marker="o",
            lw=2.0,
            capsize=3,
            label=label,
            color=colors.get(label, "#464C55"),
        )
    ax.axhline(30.0, color="#9AA1B2", lw=1.0, ls=":", label="30 s horizon")
    ax.set_ylabel("final time [s]")
    ax.set_xlabel("yaw-rate amplitude [rad/s]")
    ax.set_ylim(0.0, 31.5)
    ax.set_title("Experiment 2 final-time turning range", loc="left", fontsize=13, pad=20)
    ax.text(
        0.0,
        1.02,
        "Mean final time over repeated trials; error bars indicate one standard deviation.",
        transform=ax.transAxes,
        fontsize=9,
        color="#6F768A",
    )
    ax.grid(True, alpha=0.6)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_task_valid_range.png", dpi=180)
    fig.savefig(FIGURE_DIR / f"exp2_task_valid_range_{out_dir.name}.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
    metrics = [
        ("mean_rms_yaw_err", "yaw RMS [rad]"),
        ("mean_rms_wz_err", "wz RMS [rad/s]"),
        ("mean_wz_phase_lag_s", "wz phase lag [s]"),
        ("mean_wz_high_freq_residual_rms", "wz residual RMS [rad/s]"),
    ]
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for label in sorted({str(r["controller_label"]) for r in summary}):
            group = [r for r in summary if str(r["controller_label"]) == label]
            group.sort(key=lambda r: float(r["wz_amp"]))
            ax.plot(
                [float(r["wz_amp"]) for r in group],
                [float(r[metric]) for r in group],
                marker="o",
                lw=1.7,
                label=label,
                color=colors.get(label, "#464C55"),
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.6)
    axes[1, 0].set_xlabel("yaw-rate amplitude [rad/s]")
    axes[1, 1].set_xlabel("yaw-rate amplitude [rad/s]")
    axes[0, 0].set_title("Experiment 2 response quality", loc="left", fontsize=13, pad=20)
    axes[0, 0].text(
        0.0,
        1.02,
        "Metrics use the post-transient stable window and exclude post-fall divergence.",
        transform=axes[0, 0].transAxes,
        fontsize=9,
        color="#6F768A",
    )
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "exp2_response_metrics.png", dpi=180)
    fig.savefig(FIGURE_DIR / f"exp2_response_metrics_{out_dir.name}.png", dpi=180)
    plt.close(fig)


def paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    amps = sorted({float(r["wz_amp"]) for r in rows})
    for amp in amps:
        by_controller = {
            str(label): [r for r in rows if float(r["wz_amp"]) == amp and str(r["controller_label"]) == str(label)]
            for label in {r["controller_label"] for r in rows}
        }
        if "SRBM" not in by_controller or "VICM-Ac" not in by_controller:
            continue
        srbm = by_controller["SRBM"]
        vicm = by_controller["VICM-Ac"]
        output.append(
            {
                "wz_amp": amp,
                "srbm_valid_rate": mean([float(r["task_valid"]) for r in srbm]),
                "vicm_valid_rate": mean([float(r["task_valid"]) for r in vicm]),
                "srbm_success_rate": 1.0 - mean([float(r["fall"]) for r in srbm]),
                "vicm_success_rate": 1.0 - mean([float(r["fall"]) for r in vicm]),
                "vicm_minus_srbm_final_time": mean([float(r["final_time"]) for r in vicm])
                - mean([float(r["final_time"]) for r in srbm]),
                "srbm_wz_minus_vicm_wz": mean([float(r["rms_wz_err"]) for r in srbm])
                - mean([float(r["rms_wz_err"]) for r in vicm]),
                "srbm_yaw_minus_vicm_yaw": mean([float(r["rms_yaw_err"]) for r in srbm])
                - mean([float(r["rms_yaw_err"]) for r in vicm]),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-scale", type=float, default=1.7)
    parser.add_argument("--wz-amps", default="0.05 0.10 0.15 0.20 0.25 0.30")
    parser.add_argument("--wz-period", type=float, default=4.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--controllers", default="srbm,vicm_ac")
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
    parser.add_argument("--mpc-l-diag", default=MPC_L_DIAG_MAIN)
    parser.add_argument("--max-yaw-rms", type=float, default=0.12)
    parser.add_argument("--max-wz-rms", type=float, default=0.65)
    parser.add_argument("--max-path-rms", type=float, default=999.0)
    parser.add_argument("--max-torso", type=float, default=0.15)
    parser.add_argument("--representative-amp", type=float, default=0.20)
    args = parser.parse_args()

    controllers = parse_controller_list(args.controllers)
    amps = parse_float_list(args.wz_amps)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / f"exp2_turn_task_valid_lam{token(args.lambda_scale)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    write_csv(out_dir / "metadata.csv", [vars(args)])

    trial_rows: list[dict[str, object]] = []
    trials_path = out_dir / "trials.csv"
    representative: dict[str, Path] = {}

    for amp in amps:
        print(f"=== exp2 wz_amp={amp:.3f} ===", flush=True)
        for controller in controllers:
            for rep in range(1, args.repeats + 1):
                label = CONTROLLER_LABELS[controller]
                case = f"exp2_lam{token(args.lambda_scale)}_amp{token(amp)}_{controller}_r{rep}"
                env, pred_lf = make_env(
                    exp_id=2,
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
                    sine_wz_amp=amp,
                    sine_wz_period=args.wz_period,
                    sine_wz_start=args.sine_wz_start,
                    gait_switch_threshold=args.gait_switch_threshold,
                )
                start = time.monotonic()
                paths = run_trial(out_dir=out_dir, exp_id=2, case=case, env=env, pred_leg_fraction=pred_lf)
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
                    "wz_amp": amp,
                    "wz_period": args.wz_period,
                    "rep": rep,
                    **metrics,
                    "wall_time_s": wall_time,
                    "trace_path": str(paths.trace.relative_to(out_dir)),
                    "pred_path": str(paths.pred.relative_to(out_dir)) if paths.pred else "",
                    "log_path": str(paths.log.relative_to(out_dir)),
                }
                append_csv(trials_path, row, TRIAL_FIELDS)
                trial_rows.append(row)
                if rep == 1 and abs(amp - args.representative_amp) < 1e-9:
                    representative[label] = paths.trace
                print(
                    f"{case}: final={float(metrics['final_time']):.3f}s "
                    f"fall={metrics['fall']} valid={metrics['task_valid']} "
                    f"wz_rms={float(metrics['rms_wz_err']):.3f} "
                    f"yaw_rms={float(metrics['rms_yaw_err']):.3f}",
                    flush=True,
                )

    plot_summary(out_dir, trial_rows)
    write_csv(out_dir / "paired_summary.csv", paired_summary(trial_rows))
    if representative:
        plot_tracking(
            representative,
            out_dir / f"exp2_tracking_amp{token(args.representative_amp)}.png",
            title=f"Experiment 2 representative turning response, wz_amp={args.representative_amp:.2f}",
            subtitle=f"lambda={args.lambda_scale:.2f}, smoothed 0.20 s; raw post-fall divergence is excluded from metrics",
            t_min=args.sine_wz_start,
            t_max=min(args.sim_end, args.sine_wz_start + 12.0),
        )
        plot_xy_tracking(
            representative,
            out_dir / f"exp2_xy_tracking_amp{token(args.representative_amp)}.png",
            title=f"Experiment 2 representative XY trajectory, wz_amp={args.representative_amp:.2f}",
            subtitle="Nominal XY reference is integrated from logged reference velocities; it is for visualization only.",
            t_min=args.sine_wz_start,
            t_max=min(args.sim_end, args.sine_wz_start + 12.0),
        )
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
