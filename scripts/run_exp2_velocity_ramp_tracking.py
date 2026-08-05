#!/usr/bin/env python3
"""Rerun representative tracking cases from the Experiment 1 sweep."""

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
    "target_vx",
    "tswing",
    "sine_wz_amp",
    "sine_wz_period",
    "sine_wz_start",
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
    trace_paths: dict[float, dict[str, Path]],
    out_path: Path,
    *,
    title: str,
    subtitle: str,
    t_min: float,
    t_max: float,
    time_origin: float,
    smooth_s: float = 0.8,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    colors = {"SRBM": "#5477C4", "VICM-Ac": "#CC6F47"}
    lambda_values = sorted(trace_paths)
    fig, axes = plt.subplots(len(lambda_values), 1, figsize=(11.5, 3.2 * len(lambda_values)), sharex=True)
    if len(lambda_values) == 1:
        axes = [axes]
    for ax, lambda_scale in zip(axes, lambda_values):
        ref_plotted = False
        for label, path in trace_paths[lambda_scale].items():
            rows = [r for r in read_rows(path) if t_min <= parse_float(r, "time") <= t_max]
            if not rows:
                continue
            time_values = [parse_float(r, "time") - time_origin for r in rows]
            dt = max(1e-3, sum(b - a for a, b in zip(time_values[:-1], time_values[1:])) / max(1, len(time_values) - 1))
            win = max(1, int(round(smooth_s / dt)))
            vel = moving_average([parse_float(r, "vx") for r in rows], win)
            ref = [parse_float(r, "vx_ref") for r in rows]
            ax.plot(time_values, vel, lw=1.8, color=colors.get(label, "#464C55"), label=label)
            if not ref_plotted:
                ax.plot(time_values, ref, lw=1.3, ls=":", color="#464C55", label="reference")
                ref_plotted = True
        ax.set_ylabel("forward velocity vx [m/s]")
        ax.text(0.02, 0.88, rf"$\lambda={lambda_scale:.1f}$", transform=ax.transAxes)
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
    parser.add_argument("--lambda-scales", default="1.0,1.7")
    parser.add_argument("--controllers", default="srbm,vicm_ac")
    parser.add_argument("--tracking-duration", type=float, default=30.0)
    parser.add_argument("--vx", type=float, default=1.5)
    parser.add_argument("--sine-wz-amp", type=float, default=0.4)
    parser.add_argument("--sine-wz-period", type=float, default=4.0)
    parser.add_argument("--sine-wz-start", type=float, default=4.0)
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
    sim_end = args.tracking_duration

    controllers = parse_controller_list(args.controllers)
    lambda_values = [float(value) for value in args.lambda_scales.replace(",", " ").split()]
    if not lambda_values:
        raise ValueError("at least one lambda scale is required")
    lambda_tag = "_".join(f"lam{token(value)}" for value in lambda_values)
    out_dir = Path("record") / f"exp1_representative_tracking_{lambda_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.csv"
    trial_rows: list[dict[str, object]] = []
    traces: dict[float, dict[str, Path]] = {value: {} for value in lambda_values}
    for lambda_scale in lambda_values:
        print(
            f"=== exp1 representative tracking lambda={lambda_scale:.3f}: "
            f"vx={args.vx:.3f}, wz_amp={args.sine_wz_amp:.3f} ===",
            flush=True,
        )
        for controller in controllers:
            label = CONTROLLER_LABELS[controller]
            case = f"exp2_lam{token(lambda_scale)}_vx_ramp_{controller}"
            env, _ = make_env(
                exp_id=2,
                case=case,
                controller=controller,
                sim_end=sim_end,
                vx=args.vx,
                vy=0.0,
                tswing=args.tswing,
                posrot_att_scale=args.posrot_att_scale,
                posrot_pos_scale=args.posrot_pos_scale,
                tau_bias_scale=args.tau_bias_scale,
                tau_non_norm_limit=args.tau_non_norm_limit,
                ig_dot_filter_tau=args.ig_dot_filter_tau,
                mpc_l_diag=args.mpc_l_diag,
                torque_limit_scale=args.torque_limit_scale,
                walk_leg_pd_scale=args.walk_leg_pd_scale,
                lambda_scale=lambda_scale,
                sine_turn=abs(args.sine_wz_amp) > 1e-12,
                sine_wz_amp=args.sine_wz_amp,
                sine_wz_period=args.sine_wz_period,
                sine_wz_start=args.sine_wz_start,
                gait_switch_threshold=args.gait_switch_threshold,
            )
            start = time.monotonic()
            paths = run_trial(out_dir=out_dir, exp_id=2, case=case, env=env, pred_leg_fraction=0.5)
            wall_time = time.monotonic() - start
            metrics = velocity_metrics(
                paths.trace,
                "vx",
                eval_start=0.0,
                eval_end=sim_end,
            )
            row = {
                "case": case,
                "axis": "vx",
                "controller": controller,
                "controller_label": label,
                "lambda_scale": lambda_scale,
                "target_vx": args.vx,
                "tswing": args.tswing,
                "sine_wz_amp": args.sine_wz_amp,
                "sine_wz_period": args.sine_wz_period,
                "sine_wz_start": args.sine_wz_start,
                **metrics,
                "wall_time_s": wall_time,
                "trace_path": str(paths.trace.relative_to(out_dir)),
                "log_path": str(paths.log.relative_to(out_dir)),
            }
            append_csv(trials_path, row, TRIAL_FIELDS)
            trial_rows.append(row)
            traces[lambda_scale][label] = paths.trace
            print(
                f"{case}: final={float(metrics['final_time']):.3f}s "
                f"fall={metrics['fall']} rms_vx={float(metrics['rms_track_err']):.3f}",
                flush=True,
            )
    write_csv(
        out_dir / "summary.csv",
        aggregate(trial_rows, ["lambda_scale", "controller_label"], ["final_time", "rms_track_err"]),
    )
    fig_path = out_dir / "exp1_representative_velocity_tracking.png"
    plot_velocity_summary(
        traces,
        fig_path,
        title="Experiment 1 representative forward-velocity tracking",
        subtitle=(
            f"Experiment 1 conditions: vx={args.vx:.1f} m/s, "
            f"wz amplitude={args.sine_wz_amp:.2f} rad/s, "
            f"period={args.sine_wz_period:.1f} s."
        ),
        t_min=0.0,
        t_max=sim_end,
        time_origin=0.0,
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure_copy = FIGURE_DIR / f"exp1_representative_velocity_tracking_{out_dir.name}.png"
    figure_copy.write_bytes(fig_path.read_bytes())
    print(f"OUT={out_dir}", flush=True)
    print(f"FIG={fig_path}", flush=True)
    print(f"FIG_COPY={figure_copy}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
