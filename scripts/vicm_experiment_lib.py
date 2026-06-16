#!/usr/bin/env python3
"""Shared helpers for the final VICM/SRBM paper experiments."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
BIN = BUILD_DIR / "walk_mpc_wbc"
RECORD_DIR = REPO_ROOT / "record"
FIGURE_DIR = REPO_ROOT / "figures"

MPC_L_DIAG_MAIN = "50 50 80 1 200 1 1 1 10 100 10 1"

CONTROLLER_LABELS = {
    "srbm": "SRBM",
    "vicm_ig": "VICM-Ig",
    "vicm_ac": "VICM-Ac",
    "vicm_ac_nofilter": "VICM-Ac no filter",
    "vicm_affine": "VICM affine tau",
}


@dataclass
class TrialPaths:
    log: Path
    trace: Path
    datalog: Path | None
    pred: Path | None


def token(value: float | str) -> str:
    if isinstance(value, str):
        return value.replace(".", "p").replace("-", "m").replace(",", "_")
    return f"{value:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in text.replace(",", " ").split():
        if part.strip():
            values.append(float(part))
    return values


def parse_controller_list(text: str) -> list[str]:
    controllers = [c.strip().lower() for c in text.replace(" ", ",").split(",") if c.strip()]
    unknown = [c for c in controllers if c not in CONTROLLER_LABELS]
    if unknown:
        raise ValueError(f"unknown controller variant(s): {', '.join(unknown)}")
    if not controllers:
        raise ValueError("at least one controller is required")
    return controllers


def mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else math.nan


def stdev_pop(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    avg = mean(finite)
    return math.sqrt(mean([(v - avg) * (v - avg) for v in finite]))


def rms(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return math.sqrt(mean([v * v for v in finite])) if finite else math.nan


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fields if fields is not None else list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, object], fields: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def controller_env(
    variant: str,
    ig_dot_filter_tau: float,
    tau_non_norm_limit: float,
) -> dict[str, str]:
    if variant == "srbm":
        return {
            "ODC_USE_VICM": "0",
            "ODC_USE_TAU_BIAS": "0",
            "ODC_LINEAR_TAU_DYNAMICS": "0",
            "ODC_IG_DOT_FILTER_TAU": f"{ig_dot_filter_tau:.12g}",
            "ODC_TAU_NON_NORM_LIMIT": f"{tau_non_norm_limit:.12g}",
        }
    if variant == "vicm_ig":
        return {
            "ODC_USE_VICM": "1",
            "ODC_USE_TAU_BIAS": "0",
            "ODC_LINEAR_TAU_DYNAMICS": "0",
            "ODC_IG_DOT_FILTER_TAU": f"{ig_dot_filter_tau:.12g}",
            "ODC_TAU_NON_NORM_LIMIT": f"{tau_non_norm_limit:.12g}",
        }
    if variant == "vicm_ac":
        return {
            "ODC_USE_VICM": "1",
            "ODC_USE_TAU_BIAS": "1",
            "ODC_LINEAR_TAU_DYNAMICS": "1",
            "ODC_IG_DOT_FILTER_TAU": f"{ig_dot_filter_tau:.12g}",
            "ODC_TAU_NON_NORM_LIMIT": f"{tau_non_norm_limit:.12g}",
        }
    if variant == "vicm_ac_nofilter":
        return {
            "ODC_USE_VICM": "1",
            "ODC_USE_TAU_BIAS": "1",
            "ODC_LINEAR_TAU_DYNAMICS": "1",
            "ODC_IG_DOT_FILTER_TAU": "0",
            "ODC_TAU_NON_NORM_LIMIT": f"{tau_non_norm_limit:.12g}",
        }
    if variant == "vicm_affine":
        return {
            "ODC_USE_VICM": "1",
            "ODC_USE_TAU_BIAS": "1",
            "ODC_LINEAR_TAU_DYNAMICS": "0",
            "ODC_IG_DOT_FILTER_TAU": f"{ig_dot_filter_tau:.12g}",
            "ODC_TAU_NON_NORM_LIMIT": f"{tau_non_norm_limit:.12g}",
        }
    raise ValueError(f"unknown controller variant: {variant}")


def make_env(
    *,
    exp_id: int,
    case: str,
    controller: str,
    sim_end: float,
    vx: float,
    tswing: float,
    posrot_att_scale: float,
    posrot_pos_scale: float,
    tau_bias_scale: float,
    tau_non_norm_limit: float,
    ig_dot_filter_tau: float,
    mpc_l_diag: str,
    torque_limit_scale: float,
    walk_leg_pd_scale: float,
    vy: float = 0.0,
    lambda_scale: float | None = None,
    leg_mass_fraction: float = 0.5,
    sine_turn: bool = True,
    sine_wz_amp: float = 0.25,
    sine_wz_period: float = 4.0,
    sine_wz_start: float = 4.0,
    push_force: float = 0.0,
    push_start: float = 6.0,
    push_duration: float = 0.15,
    push_dir_x: float = 1.0,
    push_dir_y: float = 0.0,
    push_dir_z: float = 0.0,
    push_trigger_mode: str = "time",
    push_trigger_phi: float = 0.5,
    gait_switch_threshold: float = 100.0,
    foot_lookahead_time: float | None = None,
    wbc_delta_fr_weight: float | None = None,
    wbc_delta_ddq_weight: float | None = None,
) -> tuple[dict[str, str], float]:
    env = os.environ.copy()
    use_lambda = lambda_scale is not None
    effective_lf = leg_mass_fraction
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_EXP": str(exp_id),
            "ODC_RUN_LABEL": case,
            "ODC_USE_LEG_LAMBDA_SCALE": "1" if use_lambda else "0",
            "ODC_LEG_MASS_FRACTION": f"{leg_mass_fraction:.12g}",
            "ODC_TARGET_SPEED_X": f"{vx:.12g}",
            "ODC_TARGET_SPEED_Y": f"{vy:.12g}",
            "ODC_TSWING": f"{tswing:.12g}",
            "ODC_GAIT_SWITCH_FORCE_SOURCE": "touch",
            "ODC_GAIT_SWITCH_FORCE_THRESHOLD": f"{gait_switch_threshold:.12g}",
            "ODC_SIM_END_TIME": f"{sim_end:.12g}",
            "ODC_TAU_BIAS_SCALE": f"{tau_bias_scale:.12g}",
            "ODC_PREDICT_IG_LINEAR": "0",
            "ODC_MPC_L_DIAG": mpc_l_diag,
            "ODC_TORQUE_LIMIT_SCALE": f"{torque_limit_scale:.12g}",
            "ODC_WALK_LEG_PD_SCALE": f"{walk_leg_pd_scale:.12g}",
            "ODC_WBC_POSROT_POS_KP_SCALE": f"{posrot_pos_scale:.12g}",
            "ODC_WBC_POSROT_POS_KD_SCALE": f"{posrot_pos_scale:.12g}",
            "ODC_WBC_POSROT_ATT_KP_SCALE": f"{posrot_att_scale:.12g}",
            "ODC_WBC_POSROT_ATT_KD_SCALE": f"{posrot_att_scale:.12g}",
            "ODC_LOG_PREDICTION_ERROR": "1",
            "ODC_PRINT_MPC_TIMING": "0",
            "ODC_PRINT_FR_FF": "0",
            "ODC_PRINT_GAIT_SWITCH": "0",
            "ODC_SINE_TURN": "1" if sine_turn else "0",
            "ODC_SINE_WZ_BASE": "0",
            "ODC_SINE_WZ_AMP": f"{sine_wz_amp:.12g}",
            "ODC_SINE_WZ_PERIOD": f"{sine_wz_period:.12g}",
            "ODC_SINE_WZ_START_TIME": f"{sine_wz_start:.12g}",
            "ODC_PUSH_FORCE": f"{push_force:.12g}",
            "ODC_PUSH_START_TIME": f"{push_start:.12g}",
            "ODC_PUSH_DURATION": f"{push_duration:.12g}",
            "ODC_PUSH_DIR_X": f"{push_dir_x:.12g}",
            "ODC_PUSH_DIR_Y": f"{push_dir_y:.12g}",
            "ODC_PUSH_DIR_Z": f"{push_dir_z:.12g}",
            "ODC_PUSH_TRIGGER_MODE": push_trigger_mode,
            "ODC_PUSH_TRIGGER_PHI": f"{push_trigger_phi:.12g}",
        }
    )
    if use_lambda:
        env["ODC_LEG_LAMBDA_SCALE"] = f"{lambda_scale:.12g}"
        # The executable keeps the nominal leg fraction at 0.5 when lambda scaling
        # is used, and the generated prediction filename includes that value.
        effective_lf = leg_mass_fraction
    env.update(controller_env(controller, ig_dot_filter_tau, tau_non_norm_limit))
    if wbc_delta_fr_weight is not None:
        env["ODC_WBC_DELTA_FR_WEIGHT"] = f"{wbc_delta_fr_weight:.12g}"
    if wbc_delta_ddq_weight is not None:
        env["ODC_WBC_DELTA_DDQ_WEIGHT"] = f"{wbc_delta_ddq_weight:.12g}"
    if foot_lookahead_time is not None:
        env["ODC_FOOT_LOOKAHEAD_TIME"] = f"{foot_lookahead_time:.12g}"
    return env, effective_lf


def run_trial(
    *,
    out_dir: Path,
    exp_id: int,
    case: str,
    env: dict[str, str],
    pred_leg_fraction: float,
) -> TrialPaths:
    if not BIN.exists():
        raise FileNotFoundError(f"missing executable: {BIN}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{case}.log"
    with log_path.open("w") as log:
        subprocess.run(
            [str(BIN)],
            cwd=BUILD_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    trace_src = RECORD_DIR / f"exp{exp_id}_trace.csv"
    trace_dst = out_dir / f"{case}_trace.csv"
    shutil.copyfile(trace_src, trace_dst)

    datalog_dst: Path | None = None
    datalog_src = RECORD_DIR / f"exp{exp_id}_datalog.log"
    if datalog_src.exists():
        datalog_dst = out_dir / f"{case}_datalog.log"
        shutil.copyfile(datalog_src, datalog_dst)

    pred_dst: Path | None = None
    pred_src = RECORD_DIR / f"pred_error_exp{exp_id}_{case}_lf{pred_leg_fraction:.6f}.csv"
    if pred_src.exists():
        pred_dst = out_dir / f"{case}_pred_error.csv"
        shutil.copyfile(pred_src, pred_dst)

    return TrialPaths(log=log_path, trace=trace_dst, datalog=datalog_dst, pred=pred_dst)


def fit_sinusoid(time_values: list[float], values: list[float], period: float) -> dict[str, float]:
    paired = [(t, v) for t, v in zip(time_values, values) if math.isfinite(t) and math.isfinite(v)]
    if len(paired) < 8 or period <= 1e-9:
        return {"amp": math.nan, "phase": math.nan, "offset": math.nan, "fit_rms": math.nan}
    import numpy as np

    t = np.array([p[0] for p in paired], dtype=float)
    y = np.array([p[1] for p in paired], dtype=float)
    omega = 2.0 * math.pi / period
    a = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    sin_c, cos_c, offset = coef
    fit = a @ coef
    amp = math.sqrt(float(sin_c * sin_c + cos_c * cos_c))
    phase = math.atan2(float(cos_c), float(sin_c))
    return {
        "amp": amp,
        "phase": phase,
        "offset": float(offset),
        "fit_rms": rms((y - fit).tolist()),
    }


def phase_lag_seconds(actual_phase: float, ref_phase: float, period: float) -> float:
    if not (math.isfinite(actual_phase) and math.isfinite(ref_phase) and period > 1e-9):
        return math.nan
    delta = actual_phase - ref_phase
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return -delta / (2.0 * math.pi / period)


def compute_trace_metrics(
    trace_path: Path,
    pred_path: Path | None,
    *,
    sim_end: float,
    sine_start: float,
    sine_period: float,
    eval_start_offset: float = 1.0,
    eval_end_margin: float = 0.5,
    min_valid_duration: float = 8.0,
    max_yaw_rms: float = 0.12,
    max_wz_rms: float = 0.65,
    max_path_rms: float = 1.25,
    max_torso: float = 0.15,
) -> dict[str, float | int]:
    rows = read_rows(trace_path)
    if not rows:
        raise RuntimeError(f"empty trace: {trace_path}")
    final_time = parse_float(rows[-1], "time")
    fall = 0
    fall_time = final_time
    for row in rows:
        if parse_float(row, "fall_detected", 0.0) > 0.5:
            fall = 1
            fall_time = parse_float(row, "time")
            break
    if not fall:
        fall_time = final_time

    times = [parse_float(r, "time") for r in rows]
    x_ref: list[float] = []
    y_ref: list[float] = []
    if rows:
        x_acc = parse_float(rows[0], "base_x", 0.0)
        y_acc = parse_float(rows[0], "base_y", 0.0)
        last_t = times[0]
        for idx, row in enumerate(rows):
            t = times[idx]
            dt = max(0.0, t - last_t)
            x_acc += parse_float(row, "vx_ref", 0.0) * dt
            y_acc += parse_float(row, "vy_ref", 0.0) * dt
            x_ref.append(x_acc)
            y_ref.append(y_acc)
            last_t = t

    eval_start = sine_start + eval_start_offset
    eval_end = min(fall_time if fall else final_time, sim_end) - eval_end_margin
    eval_indices = [
        i for i, row in enumerate(rows)
        if parse_float(row, "time") >= eval_start and parse_float(row, "time") <= eval_end
    ]
    if not eval_indices:
        eval_indices = [i for i, row in enumerate(rows) if parse_float(row, "time") >= sine_start]
    if not eval_indices:
        eval_indices = list(range(len(rows)))

    ev = [rows[i] for i in eval_indices]
    ev_t = [times[i] for i in eval_indices]
    yaw = [parse_float(r, "yaw") for r in ev]
    yaw_ref = [parse_float(r, "yaw_ref") for r in ev]
    wz = [parse_float(r, "wz") for r in ev]
    wz_ref = [parse_float(r, "wz_ref") for r in ev]
    yaw_err = [a - b for a, b in zip(yaw, yaw_ref)]
    wz_err = [a - b for a, b in zip(wz, wz_ref)]
    path_err = [
        math.hypot(parse_float(rows[i], "base_x") - x_ref[i], parse_float(rows[i], "base_y") - y_ref[i])
        for i in eval_indices
    ]
    torso = [parse_float(r, "torso_angle_error") for r in ev]
    vel_err = [parse_float(r, "vel_track_error") for r in ev]
    wbc_delta = [parse_float(r, "wbc_delta_fr_norm") for r in ev]
    tau_mpc_norm = [parse_float(r, "tau_mpc_norm") for r in ev]

    wz_fit = fit_sinusoid(ev_t, wz, sine_period)
    wz_ref_fit = fit_sinusoid(ev_t, wz_ref, sine_period)
    yaw_fit = fit_sinusoid(ev_t, yaw, sine_period)
    yaw_ref_fit = fit_sinusoid(ev_t, yaw_ref, sine_period)
    wz_err_fit = fit_sinusoid(ev_t, wz_err, sine_period)

    srbm_pred_err: list[float] = []
    vicm_pred_err: list[float] = []
    if pred_path is not None and pred_path.exists():
        for row in read_rows(pred_path):
            if parse_float(row, "time") >= eval_start and parse_float(row, "time") <= eval_end:
                srbm_pred_err.append(parse_float(row, "srbm_err_norm"))
                vicm_pred_err.append(parse_float(row, "vicm_err_norm"))

    eval_duration = max(0.0, eval_end - eval_start)
    task_valid = (
        fall == 0
        and final_time >= min(sim_end - 1e-6, sine_start + min_valid_duration)
        and eval_duration >= min_valid_duration
        and rms(yaw_err) <= max_yaw_rms
        and rms(wz_err) <= max_wz_rms
        and rms(path_err) <= max_path_rms
        and max(torso, default=math.nan) <= max_torso
    )

    return {
        "fall": fall,
        "fall_time": fall_time,
        "final_time": final_time,
        "steps": int(parse_float(rows[-1], "step_count", 0.0)),
        "eval_start": eval_start,
        "eval_end": eval_end,
        "eval_duration": eval_duration,
        "task_valid": int(task_valid),
        "rms_yaw_err": rms(yaw_err),
        "max_abs_yaw_err": max((abs(v) for v in yaw_err), default=math.nan),
        "rms_wz_err": rms(wz_err),
        "max_abs_wz_err": max((abs(v) for v in wz_err), default=math.nan),
        "rms_path_err": rms(path_err),
        "max_path_err": max(path_err, default=math.nan),
        "rms_vel_err": rms(vel_err),
        "max_torso_angle_error": max(torso, default=math.nan),
        "mean_wbc_delta_fr_norm": mean(wbc_delta),
        "max_wbc_delta_fr_norm": max(wbc_delta, default=math.nan),
        "rms_tau_mpc_norm": rms(tau_mpc_norm),
        "wz_gain": wz_fit["amp"] / wz_ref_fit["amp"] if wz_ref_fit["amp"] and math.isfinite(wz_ref_fit["amp"]) else math.nan,
        "wz_phase_lag_s": phase_lag_seconds(wz_fit["phase"], wz_ref_fit["phase"], sine_period),
        "yaw_gain": yaw_fit["amp"] / yaw_ref_fit["amp"] if yaw_ref_fit["amp"] and math.isfinite(yaw_ref_fit["amp"]) else math.nan,
        "yaw_phase_lag_s": phase_lag_seconds(yaw_fit["phase"], yaw_ref_fit["phase"], sine_period),
        "wz_high_freq_residual_rms": wz_err_fit["fit_rms"],
        "rms_srbm_pred_err": rms(srbm_pred_err),
        "rms_vicm_pred_err": rms(vicm_pred_err),
    }


def aggregate(rows: list[dict[str, object]], keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: list[dict[str, object]] = []
    for key_values, group in sorted(groups.items(), key=lambda item: item[0]):
        result = {key: value for key, value in zip(keys, key_values)}
        result["n"] = len(group)
        result["fall_count"] = sum(int(float(g.get("fall", 0))) for g in group)
        result["task_valid_count"] = sum(int(float(g.get("task_valid", 0))) for g in group)
        result["success_rate"] = 1.0 - result["fall_count"] / len(group)
        result["task_valid_rate"] = result["task_valid_count"] / len(group)
        for metric in metrics:
            values = [float(g.get(metric, math.nan)) for g in group]
            result[f"mean_{metric}"] = mean(values)
            result[f"std_{metric}"] = stdev_pop(values)
            result[f"min_{metric}"] = min([v for v in values if math.isfinite(v)], default=math.nan)
            result[f"max_{metric}"] = max([v for v in values if math.isfinite(v)], default=math.nan)
        out.append(result)
    return out


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return values
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(mean(values[lo:hi]))
    return out


def setup_matplotlib() -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    except Exception:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "text.color": "#1F2430",
            "xtick.color": "#6F768A",
            "ytick.color": "#6F768A",
            "grid.color": "#E6E8F0",
            "font.family": ["DejaVu Sans", "Arial", "sans-serif"],
        }
    )


def plot_tracking(
    trace_paths: dict[str, Path],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    t_min: float,
    t_max: float,
    smooth_s: float = 0.2,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    colors = {
        "SRBM": "#5477C4",
        "VICM-Ac": "#CC6F47",
        "VICM-Ig": "#71B436",
        "VICM-Ac no filter": "#BD569B",
        "VICM affine tau": "#736422",
    }
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True)
    ref_plotted = False
    for label, path in trace_paths.items():
        rows = [r for r in read_rows(path) if t_min <= parse_float(r, "time") <= t_max]
        if not rows:
            continue
        time_values = [parse_float(r, "time") for r in rows]
        dt = max(1e-3, mean([b - a for a, b in zip(time_values[:-1], time_values[1:])]))
        win = max(1, int(round(smooth_s / dt)))
        yaw = moving_average([parse_float(r, "yaw") for r in rows], win)
        wz = moving_average([parse_float(r, "wz") for r in rows], win)
        yaw_ref = [parse_float(r, "yaw_ref") for r in rows]
        wz_ref = [parse_float(r, "wz_ref") for r in rows]
        color = colors.get(label, "#464C55")
        axes[0].plot(time_values, yaw, lw=1.7, color=color, label=label)
        axes[1].plot(time_values, wz, lw=1.7, color=color, label=label)
        if not ref_plotted:
            axes[0].plot(time_values, yaw_ref, lw=1.2, ls=":", color="#464C55", label="reference")
            axes[1].plot(time_values, wz_ref, lw=1.2, ls=":", color="#464C55", label="reference")
            ref_plotted = True
    axes[0].set_ylabel("yaw [rad]")
    axes[1].set_ylabel("wz [rad/s]")
    axes[1].set_xlabel("time [s]")
    axes[0].set_title(title, loc="left", fontsize=13, pad=20)
    axes[0].text(0.0, 1.02, subtitle, transform=axes[0].transAxes, fontsize=9, color="#6F768A")
    for ax in axes:
        ax.grid(True, alpha=0.6)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_xy_tracking(
    trace_paths: dict[str, Path],
    output_path: Path,
    *,
    title: str,
    subtitle: str,
    t_min: float,
    t_max: float,
    smooth_s: float = 0.2,
    show_reference: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    colors = {
        "SRBM": "#5477C4",
        "VICM-Ac": "#CC6F47",
        "VICM-Ig": "#71B436",
        "VICM-Ac no filter": "#BD569B",
        "VICM affine tau": "#736422",
    }
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.8))
    ref_plotted = False
    for label, path in trace_paths.items():
        rows = [r for r in read_rows(path) if t_min <= parse_float(r, "time") <= t_max]
        if not rows:
            continue
        time_values = [parse_float(r, "time") for r in rows]
        dt = max(1e-3, mean([b - a for a, b in zip(time_values[:-1], time_values[1:])]))
        win = max(1, int(round(smooth_s / dt)))
        x = moving_average([parse_float(r, "base_x") for r in rows], win)
        y = moving_average([parse_float(r, "base_y") for r in rows], win)
        color = colors.get(label, "#464C55")
        ax.plot(x, y, lw=1.8, color=color, label=label)
        ax.scatter([x[0]], [y[0]], s=28, color=color, marker="o", zorder=3)
        ax.scatter([x[-1]], [y[-1]], s=36, color=color, marker="x", zorder=3)

        if show_reference and not ref_plotted:
            ref_x = [parse_float(rows[0], "base_x")]
            ref_y = [parse_float(rows[0], "base_y")]
            for prev, row in zip(rows[:-1], rows[1:]):
                step_dt = max(0.0, parse_float(row, "time") - parse_float(prev, "time"))
                yaw_ref = parse_float(row, "yaw_ref", 0.0)
                vx_ref = parse_float(row, "vx_ref", 0.0)
                vy_ref = parse_float(row, "vy_ref", 0.0)
                world_vx = vx_ref * math.cos(yaw_ref) - vy_ref * math.sin(yaw_ref)
                world_vy = vx_ref * math.sin(yaw_ref) + vy_ref * math.cos(yaw_ref)
                ref_x.append(ref_x[-1] + world_vx * step_dt)
                ref_y.append(ref_y[-1] + world_vy * step_dt)
            ax.plot(ref_x, ref_y, lw=1.2, ls=":", color="#464C55", label="nominal XY reference")
            ref_plotted = True

    ax.set_xlabel("base x [m]")
    ax.set_ylabel("base y [m]")
    ax.set_title(title, loc="left", fontsize=13, pad=20)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#6F768A")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.margins(x=0.02, y=0.10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
