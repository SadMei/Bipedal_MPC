#!/usr/bin/env python3
"""Experiment 3: phase-aligned push-recovery region comparison."""

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
    parse_float,
    read_rows,
    run_trial,
    setup_matplotlib,
    token,
    write_csv,
)


TRIAL_FIELDS = [
    "case",
    "controller",
    "controller_label",
    "recovery_mode",
    "lambda_scale",
    "push_angle_deg",
    "push_force",
    "push_start",
    "push_duration",
    "push_trigger_mode",
    "push_trigger_phi",
    "push_actual_start",
    "recovery_stop_steps",
    "rep",
    "fall",
    "recovered",
    "recovery_horizon_reached",
    "evaluation_time",
    "stop_planar_velocity_mean_norm",
    "stop_angular_velocity_mean_norm",
    "stop_max_torso_angle",
    "recovered_at_step",
    "stopped",
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

RECOVERY_HORIZON_FIELDS = [
    "case",
    "controller",
    "controller_label",
    "push_angle_deg",
    "push_force",
    "rep",
    "horizon_steps",
    "recovery_horizon_reached",
    "evaluation_time",
    "stop_planar_velocity_mean_norm",
    "stop_angular_velocity_mean_norm",
    "stop_max_torso_angle",
    "recovered_at_step",
    "transient_stopped",
    "survived_to_sim_end",
    "sustained_recovered",
]


def recovery_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = aggregate(
        rows,
        ["controller_label", "push_angle_deg", "push_force"],
        ["final_time", "max_torso_angle_error", "rms_path_err"],
    )
    grouped: dict[tuple[str, float, float], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["controller_label"]),
            float(row["push_angle_deg"]),
            float(row["push_force"]),
        )
        grouped.setdefault(key, []).append(row)
    for row in summary:
        key = (
            str(row["controller_label"]),
            float(row["push_angle_deg"]),
            float(row["push_force"]),
        )
        group = grouped[key]
        recovered_count = sum(int(float(item.get("recovered", 0))) for item in group)
        row["recovered_count"] = recovered_count
        row["success_rate"] = recovered_count / len(group)
    return summary


def evaluate_step_horizon_stop(
    trace_path: Path,
    *,
    required_steps: int,
    linear_speed_threshold: float,
    torso_threshold: float,
    evaluation_window: float,
) -> dict[str, float | int]:
    rows = read_rows(trace_path)
    trigger = next(
        (row for row in rows if parse_float(row, "push_triggered", 0.0) > 0.5),
        None,
    )
    trigger_time = (
        parse_float(trigger, "push_actual_start") if trigger is not None else math.nan
    )
    candidates: list[dict[str, float | int]] = []
    for step in range(1, required_steps + 1):
        step_row = next(
            (
                row
                for row in rows
                if parse_float(row, "time") >= trigger_time
                and int(parse_float(row, "recovery_steps", 0.0)) >= step
            ),
            None,
        )
        if step_row is None:
            continue
        evaluation_time = parse_float(step_row, "time")
        window = [
            row
            for row in rows
            if evaluation_time - evaluation_window
            <= parse_float(row, "time")
            <= evaluation_time
        ]
        if not window:
            continue
        mean_vx = mean([parse_float(row, "vx") for row in window])
        mean_vy = mean([parse_float(row, "vy") for row in window])
        mean_wx = mean([parse_float(row, "wx") for row in window])
        mean_wy = mean([parse_float(row, "wy") for row in window])
        mean_wz = mean([parse_float(row, "wz") for row in window])
        planar_velocity_mean_norm = math.hypot(mean_vx, mean_vy)
        angular_velocity_mean_norm = math.sqrt(
            mean_wx**2 + mean_wy**2 + mean_wz**2
        )
        max_torso = max(parse_float(row, "torso_angle_error") for row in window)
        upright = all(
            parse_float(row, "fall_detected", 0.0) < 0.5 for row in window
        )
        candidates.append(
            {
                "step": step,
                "evaluation_time": evaluation_time,
                "planar_velocity": planar_velocity_mean_norm,
                "angular_velocity": angular_velocity_mean_norm,
                "max_torso": max_torso,
                "qualifies": int(
                    upright
                    and planar_velocity_mean_norm <= linear_speed_threshold
                    and max_torso <= torso_threshold
                ),
            }
        )

    if not candidates:
        return {
            "push_actual_start": trigger_time,
            "recovery_horizon_reached": 0,
            "evaluation_time": math.nan,
            "stop_planar_velocity_mean_norm": math.nan,
            "stop_angular_velocity_mean_norm": math.nan,
            "stop_max_torso_angle": math.nan,
            "recovered_at_step": 0,
            "stopped": 0,
        }

    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    selected = (
        qualifying[0]
        if qualifying
        else min(candidates, key=lambda candidate: float(candidate["planar_velocity"]))
    )
    stopped = int(bool(qualifying))
    return {
        "push_actual_start": trigger_time,
        "recovery_horizon_reached": int(
            int(candidates[-1]["step"]) >= required_steps
        ),
        "evaluation_time": selected["evaluation_time"],
        "stop_planar_velocity_mean_norm": selected["planar_velocity"],
        "stop_angular_velocity_mean_norm": selected["angular_velocity"],
        "stop_max_torso_angle": selected["max_torso"],
        "recovered_at_step": selected["step"] if stopped else 0,
        "stopped": stopped,
    }


def recovery_boundary(rows: list[dict[str, object]], threshold: float) -> list[dict[str, object]]:
    summary = recovery_summary(rows)
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


def plot_recovery(
    out_dir: Path,
    rows: list[dict[str, object]],
    threshold: float,
    recovery_mode: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    setup_matplotlib()
    summary = recovery_summary(rows)
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
    criterion = (
        "Cells show the fraction of runs that reached the simulation horizon without falling."
        if recovery_mode == "survive"
        else "Cells show the fraction of runs that stopped within the prescribed post-push steps."
    )
    fig.text(
        0.02,
        0.94,
        criterion,
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
    parser.add_argument("--forces", default="0 50 100 150 200 250 300 350 400 450")
    parser.add_argument("--directions", default="0 45 90 135 180 225 270 315")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--controllers", default="srbm,ir_cmpc")
    parser.add_argument("--sim-end", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--posrot-pos-scale", type=float, default=1.0)
    parser.add_argument("--tau-bias-scale", type=float, default=1.0)
    parser.add_argument("--tau-non-norm-limit", type=float, default=0.0)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--hrel-dot-filter-tau", type=float, default=0.01)
    parser.add_argument(
        "--hrel-contact-reset",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--push-start", type=float, default=8.0)
    parser.add_argument("--push-duration", type=float, default=0.10)
    parser.add_argument("--push-trigger-mode", choices=["time", "phase"], default="phase")
    parser.add_argument("--push-trigger-phi", type=float, default=0.5)
    parser.add_argument(
        "--recovery-mode",
        choices=["stop", "survive"],
        default="stop",
        help="Use the post-push stop criterion or the original survival-to-horizon criterion.",
    )
    parser.add_argument("--recovery-stop-steps", type=int, default=2)
    parser.add_argument("--stop-linear-speed-threshold", type=float, default=0.25)
    parser.add_argument("--stop-torso-threshold", type=float, default=0.25)
    parser.add_argument("--stop-evaluation-window", type=float, default=0.15)
    parser.add_argument("--sine-wz-amp", type=float, default=0.0)
    parser.add_argument("--sine-wz-period", type=float, default=4.0)
    parser.add_argument("--sine-wz-start", type=float, default=4.0)
    parser.add_argument("--torque-limit-scale", type=float, default=1.2)
    parser.add_argument("--walk-leg-pd-scale", type=float, default=1.2)
    parser.add_argument("--gait-switch-threshold", type=float, default=100.0)
    parser.add_argument("--mpc-l-diag", default=MPC_L_DIAG_MAIN)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Explicit result directory; defaults to a timestamped record directory.",
    )
    args = parser.parse_args()

    controllers = parse_controller_list(args.controllers)
    forces = parse_float_list(args.forces)
    directions = parse_float_list(args.directions)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "push_recovery" if args.recovery_mode == "survive" else "two_step_recovery"
    out_dir = args.output_dir or (
        RECORD_DIR / f"exp4_{mode_tag}_lam{token(args.lambda_scale)}_{stamp}"
    )
    if not out_dir.is_absolute():
        out_dir = RECORD_DIR / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    write_csv(out_dir / "metadata.csv", [vars(args)])

    rows: list[dict[str, object]] = []
    trials_path = out_dir / "trials.csv"
    horizon_path = out_dir / "recovery_horizon_metrics.csv"
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
                        hrel_dot_filter_tau=args.hrel_dot_filter_tau,
                        hrel_reset_on_contact_switch=args.hrel_contact_reset,
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
                        push_recovery_stop_steps=(
                            args.recovery_stop_steps if args.recovery_mode == "stop" else 0
                        ),
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
                    survived_to_sim_end = int(
                        metrics["fall"] == 0
                        and float(metrics["final_time"]) >= args.sim_end - 1e-3
                    )
                    if args.recovery_mode == "stop":
                        horizon_metrics: dict[int, dict[str, float | int]] = {}
                        for horizon_steps in range(2, 6):
                            horizon_metric = evaluate_step_horizon_stop(
                                paths.trace,
                                required_steps=horizon_steps,
                                linear_speed_threshold=args.stop_linear_speed_threshold,
                                torso_threshold=args.stop_torso_threshold,
                                evaluation_window=args.stop_evaluation_window,
                            )
                            horizon_metrics[horizon_steps] = horizon_metric
                            transient_stopped = int(horizon_metric["stopped"])
                            append_csv(
                                horizon_path,
                                {
                                    "case": case,
                                    "controller": controller,
                                    "controller_label": label,
                                    "push_angle_deg": angle,
                                    "push_force": force,
                                    "rep": rep,
                                    "horizon_steps": horizon_steps,
                                    **horizon_metric,
                                    "transient_stopped": transient_stopped,
                                    "survived_to_sim_end": survived_to_sim_end,
                                    "sustained_recovered": int(
                                        transient_stopped and survived_to_sim_end
                                    ),
                                },
                                RECOVERY_HORIZON_FIELDS,
                            )
                        stop_metrics = horizon_metrics[args.recovery_stop_steps]
                        recovered = int(stop_metrics["stopped"])
                    else:
                        trace_rows = read_rows(paths.trace)
                        trigger = next(
                            (
                                trace_row
                                for trace_row in trace_rows
                                if parse_float(trace_row, "push_triggered", 0.0) > 0.5
                            ),
                            None,
                        )
                        stop_metrics = {
                            "push_actual_start": (
                                parse_float(trigger, "push_actual_start")
                                if trigger is not None
                                else math.nan
                            ),
                            "recovery_horizon_reached": survived_to_sim_end,
                            "evaluation_time": float(metrics["final_time"]),
                            "stop_planar_velocity_mean_norm": math.nan,
                            "stop_angular_velocity_mean_norm": math.nan,
                            "stop_max_torso_angle": math.nan,
                            "recovered_at_step": 0,
                            "stopped": 0,
                        }
                        recovered = survived_to_sim_end
                    row = {
                        "case": case,
                        "controller": controller,
                        "controller_label": label,
                        "recovery_mode": args.recovery_mode,
                        "lambda_scale": args.lambda_scale,
                        "push_angle_deg": angle,
                        "push_force": force,
                        "push_start": args.push_start,
                        "push_duration": args.push_duration,
                        "push_trigger_mode": args.push_trigger_mode,
                        "push_trigger_phi": args.push_trigger_phi,
                        "push_actual_start": stop_metrics["push_actual_start"],
                        "recovery_stop_steps": (
                            args.recovery_stop_steps if args.recovery_mode == "stop" else 0
                        ),
                        "rep": rep,
                        **metrics,
                        **stop_metrics,
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
                        f"criterion={args.recovery_mode} "
                        f"torso_max={float(metrics['max_torso_angle_error']):.3f}",
                        flush=True,
                    )

    plot_recovery(out_dir, rows, args.boundary_threshold, args.recovery_mode)
    print(f"OUT={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
