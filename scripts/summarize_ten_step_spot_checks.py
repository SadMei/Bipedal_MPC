#!/usr/bin/env python3
"""Summarize survival and 10-step on-policy prediction for spot checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from analyze_on_policy_mpc_prediction import read_rows, summarize


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--start-time", type=float, default=4.0)
    parser.add_argument(
        "--end-time",
        type=float,
        help="Optional common evaluation end time.",
    )
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_rows: list[dict[str, object]] = []
    for directory in args.experiment_dirs:
        for trial in read_csv(directory / "trials.csv"):
            case = trial["case"]
            horizon_path = directory / f"{case}_mpc_horizon.csv"
            rows = read_rows(horizon_path)
            if not rows:
                raise RuntimeError(f"{horizon_path}: no fulfilled predictions")
            available_end = max(row["actual_time"] for row in rows)
            evaluation_end = (
                min(available_end, args.end_time)
                if args.end_time is not None
                else available_end
            )
            metrics = summarize(rows, args.start_time, evaluation_end)
            selected = next(
                (
                    row
                    for row in metrics
                    if int(row["horizon_steps"]) == args.horizon
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(
                    f"{horizon_path}: no H={args.horizon} samples in "
                    f"{args.start_time:g}--{evaluation_end:g} s"
                )
            output_rows.append(
                {
                    "lambda_scale": float(trial["lambda_scale"]),
                    "controller": trial["controller"],
                    "case": case,
                    "final_time_s": float(trial["final_time"]),
                    "fall": int(trial["fall"]),
                    "evaluation_start_s": args.start_time,
                    "evaluation_end_s": evaluation_end,
                    "horizon_steps": args.horizon,
                    "horizon_ms": 5 * args.horizon,
                    "samples": int(selected["samples"]),
                    "omega_rmse_rad_s": selected["omega_rmse"],
                    "wz_rmse_rad_s": selected["wz_rmse"],
                }
            )

    output_rows.sort(
        key=lambda row: (
            float(row["lambda_scale"]),
            str(row["controller"]),
        )
    )
    output_path = args.output or (
        args.experiment_dirs[0] / "ten_step_spot_check_summary.csv"
    )
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(
        "lambda  controller  survival  fall  samples  "
        "H10_omega_RMSE  H10_wz_RMSE"
    )
    for row in output_rows:
        print(
            f"{float(row['lambda_scale']):4.1f}  "
            f"{str(row['controller']):10s}  "
            f"{float(row['final_time_s']):8.3f}  "
            f"{int(row['fall']):4d}  "
            f"{int(row['samples']):7d}  "
            f"{float(row['omega_rmse_rad_s']):14.6f}  "
            f"{float(row['wz_rmse_rad_s']):11.6f}"
        )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
