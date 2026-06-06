#!/usr/bin/env python3
"""Experiment 2 supplement: nominal velocity ramp tracking at lambda=1."""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from vicm_experiment_lib import (
    CONTROLLER_LABELS,
    FIGURE_DIR,
    MPC_L_DIAG_MAIN,
    aggregate,
    append_csv,
    make_env,
    moving_average,
    parse_controller_list,
    parse_float,
    read_rows,
    run_trial,
    setup_matplotlib,
    token,
    write_csv,
)


TRIAL_FIELDS = [
    "case",
    "axis",
    "controller",
    "controller_label",
    "lambda_scale",
    "v1",
    "v2",
    "step_time",
    "ramp_time",
    "fall",
    "final_time",
    "rms_track_err",
    "wall_time_s",
    "trace_path",
    "log_path",
]


def rms(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    if not finite:
        return float("nan")
    return (sum(v * v for v in finite) / len(finite)) ** 0.5


def velocity_metrics(trace_path: Path, axis: str, eval_start: float, eval_end: float) -> dict[str, float | int]:
    rows = read_rows(trace_path)
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")
    vel_col = "vx" if axis == "vx" else "vy"
    ref_col = "vx_ref" if axis == "vx" else "vy_ref"
    ev = [r for r in rows if eval_start <= parse_float(r, "time") <= eval_end]
    err = [parse_float(r, vel_col) - parse_float(r, ref_col) for r in ev]
    fall = int(any(parse_float(r, "fall_detected", 0.0) > 0.5 for r in rows))
    return {
        "fall": fall,
        "final_time": parse_float(rows[-1], "time"),
        "rms_track_err": rms(err),
    }


def plot_velocity_summary(
    trace_paths: dict[str, dict[str, Path]],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    t_min: float,
    t_max: float,
    smooth_s: float = 0.8,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    colors = {"SRBM": "#5477C4", "VICM-Ac": "#CC6F47"}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True)
    axis_specs = [
        ("vx", "forward velocity vx [m/s]"),
        ("vy", "lateral velocity vy [m/s]"),
    ]
    for ax, (axis, ylabel) in zip(axes, axis_specs):
        ref_plotted = False
        for label, path in trace_paths.get(axis, {}).items():
            rows = [r for r in read_rows(path) if t_min <= parse_float(r, "time") <= t_max]
            if not rows:
                continue
            time_values = [parse_float(r, "time") for r in rows]
            dt = max(1e-3, sum(b - a for a, b in zip(time_values[:-1], time_values[1:])) / max(1, len(time_values) - 1))
            win = max(1, int(round(smooth_s / dt)))
            vel_col = "vx" if axis == "vx" else "vy"
            ref_col = "vx_ref" if axis == "vx" else "vy_ref"
            vel = moving_average([parse_float(r, vel_col) for r in rows], win)
            ref = [parse_float(r, ref_col) for r in rows]
            ax.plot(time_values, vel, lw=1.8, color=colors.get(label, "#464C55"), label=label)
            if not ref_plotted:
                ax.plot(time_values, ref, lw=1.3, ls=":", color="#464C55", label="reference")
                ref_plotted = True
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.6)
        ax.legend(loc="best", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [s]")
    axes[0].set_title(title, loc="left", fontsize=13, pad=20)
    axes[0].text(0.0, 1.02, subtitle, transform=axes[0].transAxes, fontsize=9, color="#6F768A")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--controllers", default="srbm,vicm_ac")
    parser.add_argument("--sim-end", type=float, default=24.0)
    parser.add_argument("--step-time", type=float, default=10.0)
    parser.add_argument("--ramp-time", type=float, default=4.0)
    parser.add_argument("--forward-v1", type=float, default=1.0)
    parser.add_argument("--forward-v2", type=float, default=1.8)
    parser.add_argument("--lateral-v1", type=float, default=0.15)
    parser.add_argument("--lateral-v2", type=float, default=0.30)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--posrot-att-scale", type=float, default=0.35)
    parser.add_argument("--posrot-pos-scale", type=float, default=1.0)
    parser.add_argument("--tau-bias-scale", type=float, default=1.0)
    parser.add_argument("--tau-non-norm-limit", type=float, default=0.0)
    parser.add_argument("--ig-dot-filter-tau", type=float, default=0.01)
    parser.add_argument("--torque-limit-scale", type=float, default=1.2)
    parser.add_argument("--walk-leg-pd-scale", type=float, default=1.2)
    parser.add_argument("--gait-switch-threshold", type=float, default=100.0)
    parser.add_argument("--mpc-l-diag", default=MPC_L_DIAG_MAIN)
    args = parser.parse_args()

    controllers = parse_controller_list(args.controllers)
    out_dir = Path("record") / f"exp2_velocity_ramp_lam{token(args.lambda_scale)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.csv"
    trial_rows: list[dict[str, object]] = []
    traces: dict[str, dict[str, Path]] = {"vx": {}, "vy": {}}
    specs = [
        ("vx", args.forward_v1, args.forward_v2, args.forward_v1, 0.0, args.forward_v2, 0.0),
        ("vy", args.lateral_v1, args.lateral_v2, 0.0, args.lateral_v1, 0.0, args.lateral_v2),
    ]
    for axis, v1, v2, vx1, vy1, vx2, vy2 in specs:
        print(f"=== exp2 velocity ramp {axis}: {v1:.3f}->{v2:.3f} ===", flush=True)
        for controller in controllers:
            label = CONTROLLER_LABELS[controller]
            case = f"exp2_lam{token(args.lambda_scale)}_{axis}_ramp_{controller}"
            env, _ = make_env(
                exp_id=2,
                case=case,
                controller=controller,
                sim_end=args.sim_end,
                vx=vx1,
                vy=vy1,
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
                sine_turn=False,
                sine_wz_amp=0.0,
                gait_switch_threshold=args.gait_switch_threshold,
            )
            env.update(
                {
                    "ODC_STEP_SPEED": "1",
                    "ODC_STEP_SPEED_TIME": f"{args.step_time:.12g}",
                    "ODC_STEP_SPEED_RAMP_TIME": f"{args.ramp_time:.12g}",
                    "ODC_STEP_VX_1": f"{vx1:.12g}",
                    "ODC_STEP_VX_2": f"{vx2:.12g}",
                    "ODC_STEP_VY_1": f"{vy1:.12g}",
                    "ODC_STEP_VY_2": f"{vy2:.12g}",
                }
            )
            start = time.monotonic()
            paths = run_trial(out_dir=out_dir, exp_id=2, case=case, env=env, pred_leg_fraction=0.5)
            wall_time = time.monotonic() - start
            metrics = velocity_metrics(paths.trace, axis, eval_start=4.0, eval_end=args.sim_end)
            row = {
                "case": case,
                "axis": axis,
                "controller": controller,
                "controller_label": label,
                "lambda_scale": args.lambda_scale,
                "v1": v1,
                "v2": v2,
                "step_time": args.step_time,
                "ramp_time": args.ramp_time,
                **metrics,
                "wall_time_s": wall_time,
                "trace_path": str(paths.trace.relative_to(out_dir)),
                "log_path": str(paths.log.relative_to(out_dir)),
            }
            append_csv(trials_path, row, TRIAL_FIELDS)
            trial_rows.append(row)
            traces[axis][label] = paths.trace
            print(
                f"{case}: final={float(metrics['final_time']):.3f}s "
                f"fall={metrics['fall']} rms_{axis}={float(metrics['rms_track_err']):.3f}",
                flush=True,
            )
    write_csv(out_dir / "summary.csv", aggregate(trial_rows, ["axis", "controller_label"], ["final_time", "rms_track_err"]))
    fig_path = out_dir / "exp2_velocity_ramp_tracking_lam1.png"
    plot_velocity_summary(
        traces,
        fig_path,
        title="Experiment 2 velocity ramp tracking, lambda=1",
        subtitle=(
            f"Forward: {args.forward_v1:.1f}->{args.forward_v2:.1f} m/s, "
            f"lateral: {args.lateral_v1:.2f}->{args.lateral_v2:.2f} m/s; "
            f"ramp time {args.ramp_time:.1f} s; actual velocities smoothed over 0.80 s."
        ),
        t_min=4.0,
        t_max=args.sim_end,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_copy = FIGURE_DIR / f"exp2_velocity_ramp_tracking_lam1_{out_dir.name}.png"
    figure_copy.write_bytes(fig_path.read_bytes())
    print(f"OUT={out_dir}", flush=True)
    print(f"FIG={fig_path}", flush=True)
    print(f"FIG_COPY={figure_copy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
