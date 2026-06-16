#!/usr/bin/env python3
"""Experiment 4: push-recovery region for SRBM vs VICM."""

from __future__ import annotations

import argparse
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
    "push_angle_deg",
    "push_force",
    "push_start",
    "push_duration",
    "push_trigger_mode",
    "push_trigger_phi",
    "rep",
    "fall",
    "recovered",
    "fall_time",
    "final_time",
    "steps",
    "max_torso_angle_error",
    "rms_path_err",
    "rms_vel_err",
    "mean_wbc_delta_fr_norm",
    "max_wbc_delta_fr_norm",
    "wall_time_s",
    "trace_path",
    "log_path",
]


def recovery_boundary(rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    summary = aggregate(
        rows,
        ["controller_label", "push_angle_deg", "push_force"],
        ["final_time", "max_torso_angle_error", "rms_path_err"],
    )
    by_key: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in summary:
        by_key.setdefault((str(row["controller_label"]), float(row["push_angle_deg"])), []).append(row)
    out: list[dict[str, object]] = []
    for (label, angle), group in sorted(by_key.items()):
        group.sort(key=lambda r: float(r["push_force"]))
        max_force = math.nan
        for row in group:
            if float(row["success_rate"]) >= threshold:
                max_force = float(row["push_force"])
        out.append(
            {
                "controller_label": label,
                "push_angle_deg": angle,
                "max_recoverable_force": max_force,
                "threshold": threshold,
            }
        )
    return out


def plot_recovery(out_dir: Path, rows: list[dict[str, object]], threshold: float) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    setup_matplotlib()
    summary = aggregate(
        rows,
        ["controller_label", "push_angle_deg", "push_force"],
        ["final_time", "max_torso_angle_error", "rms_path_err"],
    )
    write_csv(out_dir / "summary.csv", summary)
    boundary = recovery_boundary(rows, threshold)
    write_csv(out_dir / "recovery_boundary.csv", boundary)

    controllers = sorted({str(r["controller_label"]) for r in summary})
    directions = sorted({float(r["push_angle_deg"]) for r in summary})
    forces = sorted({float(r["push_force"]) for r in summary})

    fig, axes = plt.subplots(
        1,
        len(controllers),
        figsize=(6.0 * len(controllers), 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, label in zip(axes.ravel(), controllers):
        matrix = np.full((len(directions), len(forces)), np.nan)
        for i, angle in enumerate(directions):
            for j, force in enumerate(forces):
                matches = [
                    r for r in summary
                    if str(r["controller_label"]) == label
                    and abs(float(r["push_angle_deg"]) - angle) < 1e-9
                    and abs(float(r["push_force"]) - force) < 1e-9
                ]
                if matches:
                    matrix[i, j] = float(matches[0]["success_rate"])
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="YlGnBu", origin="lower")
        ax.set_xticks(range(len(forces)), [f"{f:.0f}" for f in forces])
        ax.set_yticks(range(len(directions)), [f"{d:.0f}" for d in directions])
        ax.set_xlabel("push force [N]")
        ax.set_ylabel("push direction [deg]")
        ax.set_title(label)
        for i in range(len(directions)):
            for j in range(len(forces)):
                if math.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=8, color="#1F2430")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="recovery rate")
    fig.suptitle("Experiment 4 push-recovery map", x=0.02, ha="left", fontsize=13)
    fig.text(
        0.02,
        0.94,
        "Cells show fraction of runs that stayed upright until the simulation end.",
        fontsize=9,
        color="#6F768A",
    )
    fig.savefig(out_dir / "exp4_recovery_heatmap.png", dpi=180)
    fig.savefig(FIGURE_DIR / f"exp4_recovery_heatmap_{out_dir.name}.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.0, 6.6))
    ax = fig.add_subplot(111, projection="polar")
    colors = {"SRBM": "#5477C4", "VICM-Ac": "#CC6F47"}
    for label in controllers:
        group = [r for r in boundary if str(r["controller_label"]) == label]
        group.sort(key=lambda r: float(r["push_angle_deg"]))
        angles = [math.radians(float(r["push_angle_deg"])) for r in group]
        forces_boundary = [0.0 if not math.isfinite(float(r["max_recoverable_force"])) else float(r["max_recoverable_force"]) for r in group]
        if angles:
            angles.append(angles[0])
            forces_boundary.append(forces_boundary[0])
        ax.plot(angles, forces_boundary, marker="o", lw=1.8, label=label, color=colors.get(label, "#464C55"))
        ax.fill(angles, forces_boundary, alpha=0.10, color=colors.get(label, "#464C55"))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_title("Maximum recoverable push force by direction", loc="left", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10), fontsize=8)
    ax.grid(True, alpha=0.6)
    fig.text(
        0.08,
        0.94,
        f"Boundary uses recovery-rate threshold {threshold:.0%}; larger radius is better.",
        fontsize=9,
        color="#6F768A",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "exp4_recovery_polar_boundary.png", dpi=180)
    fig.savefig(FIGURE_DIR / f"exp4_recovery_polar_boundary_{out_dir.name}.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-scale", type=float, default=1.7)
    parser.add_argument("--forces", default="0 100 200 300 400")
    parser.add_argument("--directions", default="0 45 90 135 180 225 270 315")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--controllers", default="srbm,vicm_ac")
    parser.add_argument("--sim-end", type=float, default=15.0)
    parser.add_argument("--vx", type=float, default=1.2)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--posrot-pos-scale", type=float, default=1.0)
    parser.add_argument("--tau-bias-scale", type=float, default=1.0)
    parser.add_argument("--tau-non-norm-limit", type=float, default=0.0)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--push-start", type=float, default=8.0)
    parser.add_argument("--push-duration", type=float, default=0.15)
    parser.add_argument("--push-trigger-mode", choices=["time", "phase"], default="phase")
    parser.add_argument("--push-trigger-phi", type=float, default=0.5)
    parser.add_argument("--sine-wz-amp", type=float, default=0.0)
    parser.add_argument("--sine-wz-period", type=float, default=4.0)
    parser.add_argument("--sine-wz-start", type=float, default=4.0)
    parser.add_argument("--torque-limit-scale", type=float, default=1.2)
    parser.add_argument("--walk-leg-pd-scale", type=float, default=1.2)
    parser.add_argument("--gait-switch-threshold", type=float, default=100.0)
    parser.add_argument("--mpc-l-diag", default=MPC_L_DIAG_MAIN)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    args = parser.parse_args()

    controllers = parse_controller_list(args.controllers)
    forces = parse_float_list(args.forces)
    directions = parse_float_list(args.directions)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RECORD_DIR / f"exp4_push_recovery_lam{token(args.lambda_scale)}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    write_csv(out_dir / "metadata.csv", [vars(args)])

    rows: list[dict[str, object]] = []
    trials_path = out_dir / "trials.csv"
    sine_turn = abs(args.sine_wz_amp) > 1e-12

    for angle in directions:
        rad = math.radians(angle)
        push_x = math.cos(rad)
        push_y = math.sin(rad)
        print(f"=== exp4 direction={angle:.0f} deg ===", flush=True)
        for force in forces:
            for controller in controllers:
                label = CONTROLLER_LABELS[controller]
                for rep in range(1, args.repeats + 1):
                    case = (
                        f"exp4_lam{token(args.lambda_scale)}_dir{token(angle)}_"
                        f"F{token(force)}_{controller}_r{rep}"
                    )
                    env, pred_lf = make_env(
                        exp_id=4,
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
                        sine_turn=sine_turn,
                        sine_wz_amp=args.sine_wz_amp,
                        sine_wz_period=args.sine_wz_period,
                        sine_wz_start=args.sine_wz_start,
                        push_force=force,
                        push_start=args.push_start,
                        push_duration=args.push_duration,
                        push_dir_x=push_x,
                        push_dir_y=push_y,
                        push_dir_z=0.0,
                        push_trigger_mode=args.push_trigger_mode,
                        push_trigger_phi=args.push_trigger_phi,
                        gait_switch_threshold=args.gait_switch_threshold,
                    )
                    start = time.monotonic()
                    paths = run_trial(out_dir=out_dir, exp_id=4, case=case, env=env, pred_leg_fraction=pred_lf)
                    wall_time = time.monotonic() - start
                    metrics = compute_trace_metrics(
                        paths.trace,
                        paths.pred,
                        sim_end=args.sim_end,
                        sine_start=args.push_start,
                        sine_period=args.sine_wz_period,
                        eval_start_offset=0.0,
                        eval_end_margin=0.0,
                        max_yaw_rms=999.0,
                        max_wz_rms=999.0,
                        max_path_rms=999.0,
                        max_torso=999.0,
                    )
                    recovered = int(metrics["fall"] == 0)
                    row = {
                        "case": case,
                        "controller": controller,
                        "controller_label": label,
                        "lambda_scale": args.lambda_scale,
                        "push_angle_deg": angle,
                        "push_force": force,
                        "push_start": args.push_start,
                        "push_duration": args.push_duration,
                        "push_trigger_mode": args.push_trigger_mode,
                        "push_trigger_phi": args.push_trigger_phi,
                        "rep": rep,
                        **metrics,
                        "recovered": recovered,
                        "wall_time_s": wall_time,
                        "trace_path": str(paths.trace.relative_to(out_dir)),
                        "log_path": str(paths.log.relative_to(out_dir)),
                    }
                    append_csv(trials_path, row, TRIAL_FIELDS)
                    rows.append(row)
                    print(
                        f"{case}: final={float(metrics['final_time']):.3f}s "
                        f"fall={metrics['fall']} recovered={recovered} "
                        f"torso_max={float(metrics['max_torso_angle_error']):.3f}",
                        flush=True,
                    )

    plot_recovery(out_dir, rows, args.boundary_threshold)
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
