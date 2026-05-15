#!/usr/bin/env python3
"""Run Experiment 1 speed/mass sweep for SRBM and VICM+tau.

The C++ demo writes fixed runtime files in record/.  This runner executes each
trial sequentially, then archives the overwritten trace/log files into a
timestamped output directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
RECORD_DIR = REPO_ROOT / "record"
EXECUTABLE = BUILD_DIR / "walk_mpc_wbc"


@dataclass(frozen=True)
class Controller:
    name: str
    use_vicm: bool
    use_tau: bool


@dataclass(frozen=True)
class Condition:
    controller: Controller
    vx: float
    leg_mass_fraction: float
    trial: int
    tswing: float
    sim_end_time: float

    @property
    def run_label(self) -> str:
        return (
            f"exp1_{self.controller.name}_t{fmt_token(self.tswing)}_"
            f"lf{fmt_token(self.leg_mass_fraction)}_"
            f"vx{fmt_token(self.vx)}_trial{self.trial}"
        )


def fmt_token(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    index = 0
    while True:
        value = round(start + index * step, 10)
        if value > stop + 1e-9:
            break
        values.append(round(value, 6))
        index += 1
    return values


def ensure_build(skip_build: bool) -> None:
    if skip_build and EXECUTABLE.exists():
        return
    subprocess.run(["cmake", "-S", ".", "-B", "build"], cwd=REPO_ROOT, check=True)
    subprocess.run(["cmake", "--build", "build", "-j4"], cwd=REPO_ROOT, check=True)


def build_plan(args: argparse.Namespace) -> list[Condition]:
    controllers = [
        Controller("SRBM", False, False),
        Controller("VICM_tau", True, True),
    ]
    speeds = [float(v) for v in args.speeds.split(",")]
    masses = frange(args.mass_start, args.mass_stop, args.mass_step)
    return [
        Condition(controller, vx, mass, trial, args.tswing, args.sim_end_time)
        for vx in speeds
        for mass in masses
        for controller in controllers
        for trial in range(1, args.repeats + 1)
    ]


def write_plan(plan: list[Condition], output_dir: Path) -> None:
    with (output_dir / "experiment_plan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_label",
                "controller",
                "use_vicm",
                "use_tau",
                "target_speed_x",
                "leg_mass_fraction",
                "trial",
                "tswing",
                "sim_end_time",
            ],
        )
        writer.writeheader()
        for cond in plan:
            writer.writerow(
                {
                    "run_label": cond.run_label,
                    "controller": cond.controller.name,
                    "use_vicm": int(cond.controller.use_vicm),
                    "use_tau": int(cond.controller.use_tau),
                    "target_speed_x": f"{cond.vx:.6f}",
                    "leg_mass_fraction": f"{cond.leg_mass_fraction:.6f}",
                    "trial": cond.trial,
                    "tswing": f"{cond.tswing:.6f}",
                    "sim_end_time": f"{cond.sim_end_time:.6f}",
                }
            )


def parse_trace(trace_path: Path) -> dict[str, float | int | str]:
    if not trace_path.exists():
        return {"parse_error": f"missing trace file: {trace_path}"}

    with trace_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"parse_error": f"empty trace file: {trace_path}"}

    fall_time = float(rows[-1]["time"])
    fall_detected = 0
    for row in rows:
        if int(float(row.get("fall_detected", 0))) != 0:
            fall_time = float(row["time"])
            fall_detected = 1
            break

    final = rows[-1]
    duration = fall_time if fall_detected else float(final["time"])
    yaw_values = [abs(float(row["yaw"])) for row in rows if row.get("yaw") not in (None, "")]
    torso_values = [
        float(row["torso_angle_error"])
        for row in rows
        if row.get("torso_angle_error") not in (None, "")
    ]
    vel_values = [
        float(row["vel_track_error"])
        for row in rows
        if row.get("vel_track_error") not in (None, "")
    ]

    return {
        "duration_s": duration,
        "fall_detected": fall_detected,
        "fall_time_s": fall_time,
        "final_time_s": float(final["time"]),
        "step_count": int(float(final["step_count"])),
        "final_base_z": float(final["base_z"]),
        "final_roll": float(final["roll"]),
        "final_pitch": float(final["pitch"]),
        "final_yaw": float(final["yaw"]),
        "max_abs_yaw": max(yaw_values) if yaw_values else float("nan"),
        "mean_torso_angle_error": statistics.mean(torso_values)
        if torso_values
        else float("nan"),
        "mean_vel_track_error": statistics.mean(vel_values)
        if vel_values
        else float("nan"),
        "parse_error": "",
    }


def run_condition(
    cond: Condition,
    index: int,
    total: int,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, str | int | float]:
    traces_dir = output_dir / "traces"
    logs_dir = output_dir / "logs"
    traces_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    trace_dst = traces_dir / f"{cond.run_label}_trace.csv"
    log_dst = logs_dir / f"{cond.run_label}.log"
    if skip_existing and trace_dst.exists():
        metrics = parse_trace(trace_dst)
        return make_raw_row(cond, 0, 0.0, str(log_dst), str(trace_dst), metrics, skipped=1)

    env = os.environ.copy()
    env.update(
        {
            "ODC_HEADLESS": "1",
            "ODC_EXP": "1",
            "ODC_TSWING": f"{cond.tswing:.12g}",
            "ODC_USE_VICM": "1" if cond.controller.use_vicm else "0",
            "ODC_USE_TAU_BIAS": "1" if cond.controller.use_tau else "0",
            "ODC_LEG_MASS_FRACTION": f"{cond.leg_mass_fraction:.12g}",
            "ODC_TARGET_SPEED_X": f"{cond.vx:.12g}",
            "ODC_TARGET_SPEED_Y": "0",
            "ODC_SIM_END_TIME": f"{cond.sim_end_time:.12g}",
            "ODC_PRINT_FR_FF": "0",
            "ODC_PRINT_MPC_TIMING": "0",
            "ODC_RUN_LABEL": cond.run_label,
        }
    )

    print(f"[{index:03d}/{total:03d}] {cond.run_label}", flush=True)
    start = time.monotonic()
    with log_dst.open("w") as log_file:
        proc = subprocess.run(
            [str(EXECUTABLE)],
            cwd=BUILD_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_time = time.monotonic() - start

    trace_src = RECORD_DIR / "exp1_trace.csv"
    if trace_src.exists():
        shutil.copy2(trace_src, trace_dst)

    metrics = parse_trace(trace_dst)
    return make_raw_row(
        cond,
        proc.returncode,
        wall_time,
        str(log_dst),
        str(trace_dst),
        metrics,
        skipped=0,
    )


def make_raw_row(
    cond: Condition,
    return_code: int,
    wall_time_s: float,
    log_path: str,
    trace_path: str,
    metrics: dict[str, float | int | str],
    skipped: int,
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "run_label": cond.run_label,
        "controller": cond.controller.name,
        "use_vicm": int(cond.controller.use_vicm),
        "use_tau": int(cond.controller.use_tau),
        "target_speed_x": f"{cond.vx:.6f}",
        "leg_mass_fraction": f"{cond.leg_mass_fraction:.6f}",
        "trial": cond.trial,
        "tswing": f"{cond.tswing:.6f}",
        "return_code": return_code,
        "wall_time_s": f"{wall_time_s:.3f}",
        "log_path": log_path,
        "trace_path": trace_path,
        "skipped": skipped,
    }
    row.update(metrics)
    return row


def write_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
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
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    groups: dict[tuple[str, str, str], list[dict[str, str | int | float]]] = {}
    for row in rows:
        if row.get("parse_error"):
            continue
        key = (
            str(row["target_speed_x"]),
            str(row["leg_mass_fraction"]),
            str(row["controller"]),
        )
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, str | int | float]] = []
    for (vx, lf, controller), group in sorted(groups.items(), key=lambda item: item[0]):
        durations = [float(row["duration_s"]) for row in group]
        falls = [int(row["fall_detected"]) for row in group]
        summary.append(
            {
                "target_speed_x": vx,
                "leg_mass_fraction": lf,
                "controller": controller,
                "n": len(group),
                "duration_mean_s": f"{statistics.mean(durations):.6f}",
                "duration_std_s": f"{statistics.pstdev(durations):.6f}"
                if len(durations) > 1
                else "0.000000",
                "duration_values_s": ";".join(f"{value:.3f}" for value in durations),
                "fall_count": sum(falls),
                "success_count": len(falls) - sum(falls),
                "success_rate": f"{(len(falls) - sum(falls)) / len(falls):.6f}",
            }
        )
    return summary


def write_report(output_dir: Path, rows: list[dict[str, str | int | float]]) -> None:
    summary = aggregate_rows(rows)
    write_rows(output_dir / "summary.csv", summary)
    lines = [
        "# Experiment 1 Speed/Mass Sweep",
        "",
        f"raw_trials: `{output_dir / 'raw_trials.csv'}`",
        f"summary: `{output_dir / 'summary.csv'}`",
        "",
        "## Quick Check",
    ]
    for row in summary:
        lines.append(
            "- "
            f"vx={row['target_speed_x']} lf={row['leg_mass_fraction']} "
            f"{row['controller']}: mean={row['duration_mean_s']}s "
            f"success={row['success_count']}/{row['n']} "
            f"values={row['duration_values_s']}"
        )
    (output_dir / "run_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speeds", default="1.5")
    parser.add_argument("--mass-start", type=float, default=0.40)
    parser.add_argument("--mass-stop", type=float, default=0.80)
    parser.add_argument("--mass-step", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tswing", type=float, default=0.45)
    parser.add_argument("--sim-end-time", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RECORD_DIR / f"exp1_speed_mass_sweep_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args)
    write_plan(plan, output_dir)
    print(f"Output directory: {output_dir}")
    print(f"Conditions: {len(plan)}")
    if args.dry_run:
        return 0

    ensure_build(args.skip_build)

    rows: list[dict[str, str | int | float]] = []
    raw_path = output_dir / "raw_trials.csv"
    for index, cond in enumerate(plan, start=1):
        row = run_condition(cond, index, len(plan), output_dir, args.skip_existing)
        rows.append(row)
        write_rows(raw_path, rows)
        write_report(output_dir, rows)

    print(f"Raw trials: {raw_path}")
    print(f"Summary: {output_dir / 'summary.csv'}")
    print(f"Report: {output_dir / 'run_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
