#!/usr/bin/env python3
"""Analyze on-policy angular-velocity prediction at the 10-step MPC horizon."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_LABELS = {
    "srbm": "SRBM",
    "vicm_ig": r"$I_G$ update",
    "vicm_ac": "IR-CMPC",
}
COLORS = {
    "srbm": "#4D4D4D",
    "vicm_ig": "#D8902F",
    "vicm_ac": "#1769AA",
}
MARKERS = {"srbm": "o", "vicm_ig": "s", "vicm_ac": "^"}


def controller_key(path: Path) -> str:
    name = path.name.lower()
    if "vicm_ac" in name:
        return "vicm_ac"
    if "vicm_ig" in name:
        return "vicm_ig"
    if "srbm" in name:
        return "srbm"
    return path.stem


def read_rows(path: Path) -> list[dict[str, float]]:
    numeric_columns = {
        "origin_time",
        "target_time",
        "actual_time",
        "horizon_steps",
        "origin_phi",
        "origin_wz_ref",
        "start_wx",
        "start_wy",
        "start_wz",
        "pred_wx",
        "pred_wy",
        "pred_wz",
        "actual_wx",
        "actual_wy",
        "actual_wz",
        "err_wx",
        "err_wy",
        "err_wz",
        "delta_wx",
        "delta_wy",
        "delta_wz",
    }
    rows: list[dict[str, float]] = []
    with path.open(newline="") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, float] = {}
            valid = True
            for column in numeric_columns:
                try:
                    value = float(source[column])
                except (KeyError, TypeError, ValueError):
                    valid = False
                    break
                if not math.isfinite(value):
                    valid = False
                    break
                row[column] = value
            if valid:
                rows.append(row)
    return rows


def summarize(
    rows: list[dict[str, float]], start: float, end: float
) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        origin_time = row["origin_time"]
        if start <= origin_time <= end:
            grouped[int(row["horizon_steps"])].append(row)

    result: list[dict[str, float]] = []
    for horizon in sorted(grouped):
        selected = grouped[horizon]
        error = np.asarray(
            [[row["err_wx"], row["err_wy"], row["err_wz"]] for row in selected]
        )
        actual_delta = np.asarray(
            [
                [row["delta_wx"], row["delta_wy"], row["delta_wz"]]
                for row in selected
            ]
        )
        error_energy = float(np.sum(error * error))
        delta_energy = float(np.sum(actual_delta * actual_delta))
        component_rmse = np.sqrt(np.mean(error * error, axis=0))
        omega_rmse = math.sqrt(error_energy / len(selected))
        delta_rms = math.sqrt(delta_energy / len(selected))
        nrmse = (
            math.sqrt(error_energy / delta_energy)
            if delta_energy > 1.0e-12
            else math.nan
        )
        skill = (
            1.0 - error_energy / delta_energy
            if delta_energy > 1.0e-12
            else math.nan
        )
        result.append(
            {
                "horizon_steps": float(horizon),
                "horizon_ms": 5.0 * horizon,
                "samples": float(len(selected)),
                "omega_rmse": omega_rmse,
                "wx_rmse": float(component_rmse[0]),
                "wy_rmse": float(component_rmse[1]),
                "wz_rmse": float(component_rmse[2]),
                "actual_delta_rms": delta_rms,
                "change_normalized_rmse": nrmse,
                "persistence_skill": skill,
            }
        )
    return result


def write_summary(
    path: Path,
    summaries: dict[str, list[dict[str, float]]],
    window_name: str,
    start: float,
    ends: dict[str, float],
) -> None:
    columns = [
        "window",
        "controller",
        "start_time",
        "end_time",
        "horizon_steps",
        "horizon_ms",
        "samples",
        "omega_rmse",
        "wx_rmse",
        "wy_rmse",
        "wz_rmse",
        "actual_delta_rms",
        "change_normalized_rmse",
        "persistence_skill",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for controller, summary in summaries.items():
            for metrics in summary:
                writer.writerow(
                    {
                        "window": window_name,
                        "controller": controller,
                        "start_time": f"{start:.6f}",
                        "end_time": f"{ends[controller]:.6f}",
                        **{
                            key: (
                                f"{value:.9g}"
                                if isinstance(value, float)
                                else value
                            )
                            for key, value in metrics.items()
                        },
                    }
                )


def make_figure(
    path: Path, summaries: dict[str, list[dict[str, float]]]
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    for controller in ("srbm", "vicm_ig", "vicm_ac"):
        if controller not in summaries:
            continue
        summary = summaries[controller]
        horizons = [row["horizon_steps"] for row in summary]
        axes[0].plot(
            horizons,
            [row["omega_rmse"] for row in summary],
            color=COLORS[controller],
            marker=MARKERS[controller],
            linewidth=1.3,
            markersize=4,
            label=DISPLAY_LABELS[controller],
        )
        axes[1].plot(
            horizons,
            [row["change_normalized_rmse"] for row in summary],
            color=COLORS[controller],
            marker=MARKERS[controller],
            linewidth=1.3,
            markersize=4,
            label=DISPLAY_LABELS[controller],
        )

    max_horizon = max(
        int(row["horizon_steps"])
        for summary in summaries.values()
        for row in summary
    )
    for axis in axes:
        axis.set_xticks(range(1, max_horizon + 1))
        axis.set_xlabel("Prediction horizon (MPC steps)")
        axis.grid(True, color="#D9D9D9", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Angular-velocity RMSE (rad s$^{-1}$)")
    axes[1].set_ylabel("Change-normalized RMSE")
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout(pad=0.7, w_pad=1.2)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--start", type=float, default=4.0)
    parser.add_argument(
        "--end-margin",
        type=float,
        default=0.25,
        help="Margin removed before the shortest trajectory endpoint.",
    )
    args = parser.parse_args()

    files = sorted(args.directory.glob("*_mpc_horizon.csv"))
    if not files:
        raise SystemExit(f"No MPC horizon logs found in {args.directory}")

    trajectories = {controller_key(path): read_rows(path) for path in files}
    valid_ends = {
        controller: max(row["actual_time"] for row in rows) - args.end_margin
        for controller, rows in trajectories.items()
    }
    common_end = min(valid_ends.values())
    common_summaries = {
        controller: summarize(rows, args.start, common_end)
        for controller, rows in trajectories.items()
    }
    own_summaries = {
        controller: summarize(rows, args.start, valid_ends[controller])
        for controller, rows in trajectories.items()
    }

    common_ends = {controller: common_end for controller in trajectories}
    write_summary(
        args.directory / "on_policy_prediction_common_window.csv",
        common_summaries,
        "common",
        args.start,
        common_ends,
    )
    write_summary(
        args.directory / "on_policy_prediction_own_window.csv",
        own_summaries,
        "own",
        args.start,
        valid_ends,
    )
    make_figure(
        args.directory / "on_policy_prediction_common_window.png",
        common_summaries,
    )

    print(f"Common comparison window: {args.start:.3f}--{common_end:.3f} s")
    max_horizon = max(
        int(row["horizon_steps"])
        for summary in common_summaries.values()
        for row in summary
    )
    print(f"{max_horizon}-step on-policy prediction:")
    for controller in ("srbm", "vicm_ig", "vicm_ac"):
        if controller not in common_summaries:
            continue
        common_final = next(
            row
            for row in common_summaries[controller]
            if int(row["horizon_steps"]) == max_horizon
        )
        own_final = next(
            row
            for row in own_summaries[controller]
            if int(row["horizon_steps"]) == max_horizon
        )
        print(
            f"  {DISPLAY_LABELS[controller]:>12}: "
            f"common RMSE={common_final['omega_rmse']:.6f}, "
            f"wz={common_final['wz_rmse']:.6f}, "
            f"normalized={common_final['change_normalized_rmse']:.3f}; "
            f"own RMSE={own_final['omega_rmse']:.6f}"
        )


if __name__ == "__main__":
    main()
