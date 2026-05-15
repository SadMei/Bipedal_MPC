#!/usr/bin/env python3
"""Run the four paper experiments and aggregate duration/error metrics.

The runner intentionally keeps the C++ demo unchanged.  Each trial is launched
through environment variables, then the overwritten runtime CSV files are copied
into a timestamped archive directory under record/.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
RECORD_DIR = REPO_ROOT / "record"
EXECUTABLE = BUILD_DIR / "walk_mpc_wbc"

NOMINAL_LEG_MASS_FRACTION = 0.38607619909502255
DEFAULT_SIM_END_TIME = 30.0
DEFAULT_REPEATS = 5
DEFAULT_TSWING = 0.45
ANALYSIS_START_TIME = 3.5


@dataclass(frozen=True)
class Controller:
    name: str
    use_vicm: bool
    use_tau: bool


@dataclass(frozen=True)
class Condition:
    exp_id: int
    exp_name: str
    controller: Controller
    trial: int
    leg_mass_fraction: float
    target_speed_x: float
    target_speed_y: float = 0.0
    push_force: float = 0.0
    push_start_time: float = 6.0
    push_duration: float = 0.15
    sim_end_time: float = DEFAULT_SIM_END_TIME

    @property
    def condition_key(self) -> str:
        return (
            f"exp{self.exp_id}|{self.exp_name}|{self.controller.name}|"
            f"lf={self.leg_mass_fraction:.6f}|vx={self.target_speed_x:.3f}|"
            f"vy={self.target_speed_y:.3f}|push={self.push_force:.3f}|"
            f"push_start={self.push_start_time:.3f}|push_dur={self.push_duration:.3f}"
        )

    @property
    def run_label(self) -> str:
        return (
            f"exp{self.exp_id}_{self.exp_name}_{self.controller.name}_"
            f"lf{fmt_token(self.leg_mass_fraction)}_vx{fmt_token(self.target_speed_x)}_"
            f"pf{fmt_token(self.push_force)}_trial{self.trial}"
        )


def fmt_token(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    i = 0
    while True:
        value = round(start + i * step, 10)
        if value > stop + 1e-9:
            break
        values.append(round(value, 3))
        i += 1
    return values


def build_plan(
    repeats: int,
    sim_end_time: float,
    dry_run: bool = False,
    exp4_push_force: float = 150.0,
) -> list[Condition]:
    srbm = Controller("SRBM", False, False)
    vicm = Controller("VICM_tau", True, True)
    srbm_tau = Controller("SRBM_tau", False, True)
    vicm_ig_only = Controller("VICM_IgOnly", True, False)

    if dry_run:
        return [
            Condition(1, "leg_fraction_sweep", srbm, 1, 0.40, 1.5, sim_end_time=sim_end_time),
            Condition(1, "leg_fraction_sweep", vicm, 1, 0.40, 1.5, sim_end_time=sim_end_time),
            Condition(2, "speed_sweep", srbm, 1, NOMINAL_LEG_MASS_FRACTION, 1.2, sim_end_time=sim_end_time),
            Condition(3, "component_ablation", srbm_tau, 1, 0.70, 1.5, sim_end_time=sim_end_time),
            Condition(
                4,
                "disturbance_recovery",
                vicm,
                1,
                NOMINAL_LEG_MASS_FRACTION,
                1.5,
                push_force=exp4_push_force,
                sim_end_time=sim_end_time,
            ),
        ]

    conditions: list[Condition] = []

    # Experiment 1: bilateral leg mass fraction sensitivity.
    for lf in frange(0.40, 0.80, 0.05):
        for controller in (srbm, vicm):
            for trial in range(1, repeats + 1):
                conditions.append(
                    Condition(
                        1,
                        "leg_fraction_sweep",
                        controller,
                        trial,
                        lf,
                        1.5,
                        sim_end_time=sim_end_time,
                    )
                )

    # Experiment 2: speed sweep at the nominal mass distribution.
    for vx in frange(1.20, 1.80, 0.10):
        for controller in (srbm, vicm):
            for trial in range(1, repeats + 1):
                conditions.append(
                    Condition(
                        2,
                        "speed_sweep",
                        controller,
                        trial,
                        NOMINAL_LEG_MASS_FRACTION,
                        vx,
                        sim_end_time=sim_end_time,
                    )
                )

    # Experiment 3: component ablation at a representative high-dynamic case.
    for controller in (srbm, srbm_tau, vicm_ig_only):
        for trial in range(1, repeats + 1):
            conditions.append(
                Condition(
                    3,
                    "component_ablation",
                    controller,
                    trial,
                    0.70,
                    1.5,
                    sim_end_time=sim_end_time,
                )
            )

    # Experiment 4: disturbance recovery at the nominal mass distribution.
    for controller in (srbm, vicm):
        for trial in range(1, repeats + 1):
            conditions.append(
                Condition(
                    4,
                    "disturbance_recovery",
                    controller,
                    trial,
                    NOMINAL_LEG_MASS_FRACTION,
                    1.5,
                    push_force=exp4_push_force,
                    sim_end_time=sim_end_time,
                )
            )

    return conditions


def ensure_build(skip_build: bool = False) -> None:
    if skip_build and EXECUTABLE.exists():
        return
    subprocess.run(["cmake", "-S", ".", "-B", "build"], cwd=REPO_ROOT, check=True)
    subprocess.run(["cmake", "--build", "build", "-j4"], cwd=REPO_ROOT, check=True)


def write_plan(plan: list[Condition], output_dir: Path) -> None:
    fields = [
        "run_label",
        "condition_key",
        "exp_id",
        "exp_name",
        "controller",
        "trial",
        "use_vicm",
        "use_tau",
        "leg_mass_fraction",
        "target_speed_x",
        "target_speed_y",
        "push_force",
        "push_start_time",
        "push_duration",
        "sim_end_time",
    ]
    with (output_dir / "experiment_plan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cond in plan:
            writer.writerow(condition_to_row(cond))


def condition_to_row(cond: Condition) -> dict[str, Any]:
    return {
        "run_label": cond.run_label,
        "condition_key": cond.condition_key,
        "exp_id": cond.exp_id,
        "exp_name": cond.exp_name,
        "controller": cond.controller.name,
        "trial": cond.trial,
        "use_vicm": int(cond.controller.use_vicm),
        "use_tau": int(cond.controller.use_tau),
        "leg_mass_fraction": f"{cond.leg_mass_fraction:.12g}",
        "target_speed_x": f"{cond.target_speed_x:.12g}",
        "target_speed_y": f"{cond.target_speed_y:.12g}",
        "push_force": f"{cond.push_force:.12g}",
        "push_start_time": f"{cond.push_start_time:.12g}",
        "push_duration": f"{cond.push_duration:.12g}",
        "sim_end_time": f"{cond.sim_end_time:.12g}",
    }


def run_condition(cond: Condition, output_dir: Path, index: int, total: int) -> dict[str, Any]:
    traces_dir = output_dir / "traces"
    fr_dir = output_dir / "fr_ff"
    logs_dir = output_dir / "logs"
    traces_dir.mkdir(parents=True, exist_ok=True)
    fr_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_EXP": str(cond.exp_id),
            "ODC_USE_VICM": "1" if cond.controller.use_vicm else "0",
            "ODC_USE_TAU_BIAS": "1" if cond.controller.use_tau else "0",
            "ODC_TSWING": f"{DEFAULT_TSWING:.12g}",
            "ODC_LEG_MASS_FRACTION": f"{cond.leg_mass_fraction:.12g}",
            "ODC_TARGET_SPEED_X": f"{cond.target_speed_x:.12g}",
            "ODC_TARGET_SPEED_Y": f"{cond.target_speed_y:.12g}",
            "ODC_PUSH_FORCE": f"{cond.push_force:.12g}",
            "ODC_PUSH_START_TIME": f"{cond.push_start_time:.12g}",
            "ODC_PUSH_DURATION": f"{cond.push_duration:.12g}",
            "ODC_SIM_END_TIME": f"{cond.sim_end_time:.12g}",
            "ODC_PRINT_IG": "0",
            "ODC_PRINT_FR_FF": "1",
            "ODC_FR_PRINT_INTERVAL": "0.05",
            "ODC_PRINT_MPC_TIMING": "0",
            "ODC_RUN_LABEL": cond.run_label,
        }
    )

    log_path = logs_dir / f"{cond.run_label}.log"
    started = time.time()
    print(
        f"[{index:03d}/{total:03d}] {cond.run_label}",
        flush=True,
    )
    with log_path.open("w") as log:
        log.write(f"# started_at={datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"# condition_key={cond.condition_key}\n")
        log.flush()
        proc = subprocess.run(
            [str(EXECUTABLE)],
            cwd=BUILD_DIR,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"# return_code={proc.returncode}\n")
        log.write(f"# wall_time_s={time.time() - started:.3f}\n")

    trace_src = RECORD_DIR / f"exp{cond.exp_id}_trace.csv"
    datalog_src = RECORD_DIR / f"exp{cond.exp_id}_datalog.log"
    fr_src = (
        RECORD_DIR
        / f"fr_ff_exp{cond.exp_id}_{cond.run_label}_lf{cond.leg_mass_fraction:.6f}.csv"
    )
    trace_dst = traces_dir / f"{cond.run_label}_trace.csv"
    datalog_dst = output_dir / "datalog" / f"{cond.run_label}_datalog.log"
    fr_dst = fr_dir / f"{cond.run_label}_fr_ff.csv"

    if trace_src.exists():
        shutil.copy2(trace_src, trace_dst)
    if fr_src.exists():
        shutil.copy2(fr_src, fr_dst)
    if os.environ.get("ODC_SAVE_DATALOG", "0") == "1" and datalog_src.exists():
        datalog_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(datalog_src, datalog_dst)

    metrics = parse_trace_metrics(trace_dst, cond)
    metrics.update(condition_to_row(cond))
    metrics.update(
        {
            "return_code": proc.returncode,
            "wall_time_s": f"{time.time() - started:.6f}",
            "trace_path": str(trace_dst.relative_to(output_dir)),
            "fr_ff_path": str(fr_dst.relative_to(output_dir)) if fr_dst.exists() else "",
            "log_path": str(log_path.relative_to(output_dir)),
        }
    )
    if not trace_dst.exists():
        metrics["parse_error"] = f"missing trace file: {trace_src}"
    return metrics


def parse_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_trace_metrics(trace_path: Path, cond: Condition) -> dict[str, Any]:
    if not trace_path.exists():
        return empty_metrics("trace_missing")

    rows: list[dict[str, str]] = []
    with trace_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return empty_metrics("trace_empty")

    times = [parse_float(r, "time") for r in rows]
    valid_times = [t for t in times if math.isfinite(t)]
    final_time = valid_times[-1] if valid_times else math.nan
    fall_flags = [parse_float(r, "fall_detected", 0.0) for r in rows]
    fall_detected = any(flag > 0.5 for flag in fall_flags)
    fall_time = final_time
    for row in rows:
        if parse_float(row, "fall_detected", 0.0) > 0.5:
            fall_time = parse_float(row, "time")
            break
    duration = fall_time if fall_detected else final_time

    analysis_rows = [r for r in rows if parse_float(r, "time") >= ANALYSIS_START_TIME]
    if not analysis_rows:
        analysis_rows = rows

    vel_errors = finite_values(parse_float(r, "vel_track_error") for r in analysis_rows)
    torso_errors = finite_values(parse_float(r, "torso_angle_error") for r in analysis_rows)
    tau_bias_norms = finite_values(parse_float(r, "tau_bias_norm") for r in analysis_rows)
    tau_mpc_norms = finite_values(parse_float(r, "tau_mpc_norm") for r in analysis_rows)
    rolls = finite_values(abs(parse_float(r, "roll")) for r in analysis_rows)
    pitches = finite_values(abs(parse_float(r, "pitch")) for r in analysis_rows)
    base_z = finite_values(parse_float(r, "base_z") for r in rows)

    metrics: dict[str, Any] = {
        "parse_error": "",
        "sample_count": len(rows),
        "duration_s": duration,
        "fall_detected": int(fall_detected),
        "fall_time_s": fall_time,
        "final_time_s": final_time,
        "mean_vel_track_error": mean_or_nan(vel_errors),
        "rms_vel_track_error": rms_or_nan(vel_errors),
        "max_vel_track_error": max_or_nan(vel_errors),
        "mean_torso_angle_error": mean_or_nan(torso_errors),
        "max_torso_angle_error": max_or_nan(torso_errors),
        "max_abs_roll": max_or_nan(rolls),
        "max_abs_pitch": max_or_nan(pitches),
        "mean_tau_bias_norm": mean_or_nan(tau_bias_norms),
        "max_tau_bias_norm": max_or_nan(tau_bias_norms),
        "mean_tau_mpc_norm": mean_or_nan(tau_mpc_norms),
        "max_tau_mpc_norm": max_or_nan(tau_mpc_norms),
        "min_base_z": min_or_nan(base_z),
        "max_base_z": max_or_nan(base_z),
    }

    if cond.exp_id == 4:
        metrics.update(parse_recovery_metrics(rows, cond))
    else:
        metrics.update(
            {
                "pre_push_torso_error": math.nan,
                "post_push_max_torso_error": math.nan,
                "recovery_time_s": math.nan,
            }
        )

    return metrics


def parse_recovery_metrics(rows: list[dict[str, str]], cond: Condition) -> dict[str, Any]:
    pre_start = max(0.0, cond.push_start_time - 1.0)
    pre_rows = [
        r
        for r in rows
        if pre_start <= parse_float(r, "time") < cond.push_start_time
    ]
    post_rows = [r for r in rows if parse_float(r, "time") >= cond.push_start_time]
    pre_errors = finite_values(parse_float(r, "torso_angle_error") for r in pre_rows)
    post_errors = finite_values(parse_float(r, "torso_angle_error") for r in post_rows)
    pre_error = mean_or_nan(pre_errors)
    max_post_error = max_or_nan(post_errors)

    threshold = max(0.05, (pre_error if math.isfinite(pre_error) else 0.0) + 0.02)
    recovery_time = math.nan
    window_s = 0.50
    for i, row in enumerate(rows):
        t = parse_float(row, "time")
        if not math.isfinite(t) or t < cond.push_start_time + cond.push_duration:
            continue
        if parse_float(row, "torso_angle_error") > threshold:
            continue
        end_t = t + window_s
        window = [
            r
            for r in rows[i:]
            if parse_float(r, "time") <= end_t
        ]
        if window and all(parse_float(r, "torso_angle_error") <= threshold for r in window):
            recovery_time = t - (cond.push_start_time + cond.push_duration)
            break

    return {
        "pre_push_torso_error": pre_error,
        "post_push_max_torso_error": max_post_error,
        "recovery_time_s": recovery_time,
    }


def empty_metrics(error: str) -> dict[str, Any]:
    keys = [
        "sample_count",
        "duration_s",
        "fall_detected",
        "fall_time_s",
        "final_time_s",
        "mean_vel_track_error",
        "rms_vel_track_error",
        "max_vel_track_error",
        "mean_torso_angle_error",
        "max_torso_angle_error",
        "max_abs_roll",
        "max_abs_pitch",
        "mean_tau_bias_norm",
        "max_tau_bias_norm",
        "mean_tau_mpc_norm",
        "max_tau_mpc_norm",
        "min_base_z",
        "max_base_z",
        "pre_push_torso_error",
        "post_push_max_torso_error",
        "recovery_time_s",
    ]
    metrics = {key: math.nan for key in keys}
    metrics["fall_detected"] = math.nan
    metrics["parse_error"] = error
    return metrics


def finite_values(values: Any) -> list[float]:
    return [value for value in values if isinstance(value, float) and math.isfinite(value)]


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def rms_or_nan(values: list[float]) -> float:
    return math.sqrt(statistics.fmean(v * v for v in values)) if values else math.nan


def max_or_nan(values: list[float]) -> float:
    return max(values) if values else math.nan


def min_or_nan(values: list[float]) -> float:
    return min(values) if values else math.nan


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key, "")) for key in fields})


def format_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.9g}"
    return value


def aggregate_trials(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["condition_key"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []

    numeric_metrics = [
        "duration_s",
        "fall_detected",
        "fall_time_s",
        "final_time_s",
        "mean_vel_track_error",
        "rms_vel_track_error",
        "max_vel_track_error",
        "mean_torso_angle_error",
        "max_torso_angle_error",
        "max_abs_roll",
        "max_abs_pitch",
        "mean_tau_bias_norm",
        "max_tau_bias_norm",
        "mean_tau_mpc_norm",
        "max_tau_mpc_norm",
        "min_base_z",
        "max_base_z",
        "pre_push_torso_error",
        "post_push_max_torso_error",
        "recovery_time_s",
        "wall_time_s",
    ]

    for key, group in groups.items():
        exp_id = int(group[0]["exp_id"])
        primary = "post_push_max_torso_error" if exp_id == 4 else "duration_s"
        valid = [row for row in group if is_finite_number(row.get(primary))]
        ordered = sorted(valid, key=lambda row: float(row[primary]))
        kept = ordered
        excluded = set()
        if len(ordered) >= 5:
            excluded = {id(ordered[0]), id(ordered[-1])}
            kept = ordered[1:-1]
        elif not kept:
            kept = group

        for row in group:
            row["outlier_excluded"] = int(id(row) in excluded)

        base = {
            "condition_key": key,
            "exp_id": group[0]["exp_id"],
            "exp_name": group[0]["exp_name"],
            "controller": group[0]["controller"],
            "use_vicm": group[0]["use_vicm"],
            "use_tau": group[0]["use_tau"],
            "leg_mass_fraction": group[0]["leg_mass_fraction"],
            "target_speed_x": group[0]["target_speed_x"],
            "target_speed_y": group[0]["target_speed_y"],
            "push_force": group[0]["push_force"],
            "push_start_time": group[0]["push_start_time"],
            "push_duration": group[0]["push_duration"],
            "trial_count": len(group),
            "kept_trial_count": len(kept),
            "primary_metric": primary,
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in kept if is_finite_number(row.get(metric))]
            base[f"{metric}_mean_trimmed"] = mean_or_nan(values)
            base[f"{metric}_std_trimmed"] = std_or_nan(values)
        summary_rows.append(base)

        representative = choose_representative_trial(kept, base, primary)
        if representative:
            rep = dict(base)
            rep.update(
                {
                    "representative_run_label": representative["run_label"],
                    "representative_trial": representative["trial"],
                    "representative_trace_path": representative["trace_path"],
                    "representative_fr_ff_path": representative.get("fr_ff_path", ""),
                    "representative_duration_s": representative.get("duration_s", ""),
                    "representative_mean_vel_track_error": representative.get(
                        "mean_vel_track_error", ""
                    ),
                    "representative_max_torso_angle_error": representative.get(
                        "max_torso_angle_error", ""
                    ),
                    "representative_recovery_time_s": representative.get("recovery_time_s", ""),
                }
            )
            representative_rows.append(rep)

    summary_rows.sort(
        key=lambda r: (
            int(r["exp_id"]),
            str(r["controller"]),
            float(r["leg_mass_fraction"]),
            float(r["target_speed_x"]),
            float(r["push_force"]),
        )
    )
    representative_rows.sort(
        key=lambda r: (
            int(r["exp_id"]),
            str(r["controller"]),
            float(r["leg_mass_fraction"]),
            float(r["target_speed_x"]),
            float(r["push_force"]),
        )
    )
    return summary_rows, representative_rows


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def std_or_nan(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def choose_representative_trial(
    rows: list[dict[str, Any]], summary: dict[str, Any], primary: str
) -> dict[str, Any] | None:
    if not rows:
        return None
    mean_primary = summary.get(f"{primary}_mean_trimmed", math.nan)
    mean_error = summary.get("mean_vel_track_error_mean_trimmed", math.nan)

    def score(row: dict[str, Any]) -> tuple[float, float, float]:
        primary_value = float(row[primary]) if is_finite_number(row.get(primary)) else math.inf
        error_value = (
            float(row["mean_vel_track_error"])
            if is_finite_number(row.get("mean_vel_track_error"))
            else math.inf
        )
        torso_value = (
            float(row["max_torso_angle_error"])
            if is_finite_number(row.get("max_torso_angle_error"))
            else math.inf
        )
        primary_delta = (
            abs(primary_value - float(mean_primary))
            if is_finite_number(mean_primary)
            else 0.0
        )
        error_delta = (
            abs(error_value - float(mean_error))
            if is_finite_number(mean_error)
            else error_value
        )
        return (primary_delta, error_delta, torso_value)

    return min(rows, key=score)


def write_readme(output_dir: Path, args: argparse.Namespace, total: int) -> None:
    text = f"""Paper experiment batch archive
created_at: {datetime.now().isoformat(timespec='seconds')}
repo_root: {REPO_ROOT}
total_trials: {total}
repeats_per_condition: {args.repeats}
sim_end_time_s: {args.sim_end_time}
exp4_push_force_N: {args.exp4_push_force}

Experiment definitions:
- Exp. 1: rho_l = 0.40..0.80, step 0.05, vx = 1.5 m/s, SRBM vs VICM_tau.
- Exp. 2: nominal rho_l = {NOMINAL_LEG_MASS_FRACTION:.12g}, vx = 1.2..1.8 m/s, step 0.1 m/s, SRBM vs VICM_tau.
- Exp. 3: rho_l = 0.70, vx = 1.5 m/s, SRBM vs SRBM_tau vs VICM_IgOnly.
- Exp. 4: nominal rho_l, vx = 1.5 m/s, horizontal push on base, SRBM vs VICM_tau.

Aggregation:
- Exp. 1--3 primary metric: duration_s.
- Exp. 4 primary metric: post_push_max_torso_error.
- With 5 repeats, the smallest and largest primary metric are excluded and the
  remaining 3 trials are averaged for paper-level visualization.

Files:
- experiment_plan.csv: complete planned trial list.
- raw_trials.csv: metrics extracted from each raw trace.
- summary_trimmed.csv: trimmed means and standard deviations per condition.
- representative_trials.csv: one kept trial selected for later error-analysis plots.
- traces/: copied expN_trace.csv for every trial.
- fr_ff/: copied feed-forward force CSV for every trial.
- logs/: stdout/stderr log for every trial.
"""
    (output_dir / "README.txt").write_text(text)


def write_sanity_check(output_dir: Path, raw_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append(f"raw_trial_count={len(raw_rows)}")
    lines.append(f"summary_condition_count={len(summary_rows)}")
    failed = [row for row in raw_rows if int(row.get("return_code", 0)) != 0]
    parse_errors = [row for row in raw_rows if row.get("parse_error")]
    lines.append(f"nonzero_return_count={len(failed)}")
    lines.append(f"parse_error_count={len(parse_errors)}")

    for exp_id in (1, 2, 3, 4):
        exp_rows = [row for row in summary_rows if int(row["exp_id"]) == exp_id]
        lines.append("")
        lines.append(f"exp{exp_id}_conditions={len(exp_rows)}")
        for row in exp_rows[:6]:
            duration = row.get("duration_s_mean_trimmed", "")
            fall = row.get("fall_detected_mean_trimmed", "")
            torso = row.get("max_torso_angle_error_mean_trimmed", "")
            recovery = row.get("recovery_time_s_mean_trimmed", "")
            lines.append(
                "  "
                f"{row['controller']} lf={row['leg_mass_fraction']} "
                f"vx={row['target_speed_x']} push={row['push_force']} "
                f"duration={format_cell(duration)} fall_rate={format_cell(fall)} "
                f"max_torso={format_cell(torso)} recovery={format_cell(recovery)}"
            )

    if failed:
        lines.append("")
        lines.append("nonzero_return_samples:")
        for row in failed[:10]:
            lines.append(f"  {row['run_label']} rc={row['return_code']} log={row['log_path']}")

    if parse_errors:
        lines.append("")
        lines.append("parse_error_samples:")
        for row in parse_errors[:10]:
            lines.append(f"  {row['run_label']} error={row['parse_error']}")

    (output_dir / "sanity_check.txt").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--sim-end-time", type=float, default=DEFAULT_SIM_END_TIME)
    parser.add_argument("--exp4-push-force", type=float, default=150.0)
    parser.add_argument("--dry-run", action="store_true", help="Run one short trial from each experiment.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    sim_end_time = args.sim_end_time
    if args.dry_run and sim_end_time == DEFAULT_SIM_END_TIME:
        sim_end_time = 0.25

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else RECORD_DIR / ("paper_experiments_dryrun_" + timestamp if args.dry_run else "paper_experiments_" + timestamp)
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    ensure_build(skip_build=args.skip_build)
    plan = build_plan(
        repeats=args.repeats,
        sim_end_time=sim_end_time,
        dry_run=args.dry_run,
        exp4_push_force=args.exp4_push_force,
    )
    write_plan(plan, output_dir)
    write_readme(output_dir, args, len(plan))

    raw_rows: list[dict[str, Any]] = []
    raw_path = output_dir / "raw_trials.csv"
    total = len(plan)
    for index, cond in enumerate(plan, start=1):
        row = run_condition(cond, output_dir, index, total)
        raw_rows.append(row)
        write_rows(raw_path, raw_rows)

    summary_rows, representative_rows = aggregate_trials(raw_rows)
    write_rows(output_dir / "raw_trials.csv", raw_rows)
    write_rows(output_dir / "summary_trimmed.csv", summary_rows)
    write_rows(output_dir / "representative_trials.csv", representative_rows)
    write_sanity_check(output_dir, raw_rows, summary_rows)

    print(f"Done. Output directory: {output_dir}")
    print(f"Summary: {output_dir / 'summary_trimmed.csv'}")
    print(f"Sanity check: {output_dir / 'sanity_check.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
