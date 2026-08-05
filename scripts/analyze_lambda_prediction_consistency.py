#!/usr/bin/env python3
"""Analyze angular-dynamics consistency across a leg-inertia lambda sweep."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_five_step_prediction_error import calculate, matrix, read_rows, vector


PREDICTORS = (
    ("SRBM", "SRBM"),
    (r"$I_G$ update", "VI-frozen"),
    (r"$I_G+\dot I_G\omega$", "IR-linear"),
)
COLORS = {
    "SRBM": "#4D4D4D",
    "VI-frozen": "#D8902F",
    "IR-linear": "#1769AA",
}
MARKERS = {"SRBM": "o", "VI-frozen": "s", "IR-linear": "^"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def last_time(path: Path) -> float:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path}: empty prediction log")
    return float(rows[-1]["time"])


def coupling_metrics(trials: list[list[dict[str, str]]]) -> dict[str, float]:
    coupling_energy = 0.0
    moment_energy = 0.0
    required_residual_energy = 0.0
    coupling_impulse_energy = 0.0
    residual_coupling_dot = 0.0
    inertias: list[np.ndarray] = []
    for rows in trials:
        for row in rows:
            dt = float(row["dt"])
            if dt <= 0.0:
                continue
            inertia_dot = matrix(row, "idot_filtered")
            omega = vector(row, "start")
            moment = np.array(
                [
                    float(row["moment_impulse_x"]),
                    float(row["moment_impulse_y"]),
                    float(row["moment_impulse_z"]),
                ]
            ) / dt
            coupling = inertia_dot @ omega
            coupling_energy += float(coupling @ coupling)
            moment_energy += float(moment @ moment)
            actual_omega = vector(row, "actual")
            angular_impulse = moment * dt
            required_residual_impulse = (
                angular_impulse - matrix(row, "inertia") @ (actual_omega - omega)
            )
            coupling_impulse = coupling * dt
            required_residual_energy += float(
                required_residual_impulse @ required_residual_impulse
            )
            coupling_impulse_energy += float(coupling_impulse @ coupling_impulse)
            residual_coupling_dot += float(
                required_residual_impulse @ coupling_impulse
            )
            inertias.append(matrix(row, "inertia"))

    coupling_ratio = (
        math.sqrt(coupling_energy / moment_energy)
        if moment_energy > 1.0e-16
        else math.nan
    )
    residual_coupling_cosine = (
        residual_coupling_dot
        / math.sqrt(required_residual_energy * coupling_impulse_energy)
        if required_residual_energy > 1.0e-16
        and coupling_impulse_energy > 1.0e-16
        else math.nan
    )
    residual_projection_scale = (
        residual_coupling_dot / coupling_impulse_energy
        if coupling_impulse_energy > 1.0e-16
        else math.nan
    )
    coupling_to_residual_ratio = (
        math.sqrt(coupling_impulse_energy / required_residual_energy)
        if required_residual_energy > 1.0e-16
        else math.nan
    )
    if not inertias:
        return {
            "coupling_ratio": coupling_ratio,
            "normalized_inertia_variation": math.nan,
            "coupling_to_required_residual_ratio": coupling_to_residual_ratio,
            "residual_coupling_cosine": residual_coupling_cosine,
            "residual_projection_scale": residual_projection_scale,
        }
    inertia_array = np.asarray(inertias)
    mean_inertia = np.mean(inertia_array, axis=0)
    denominator = float(np.linalg.norm(mean_inertia, ord="fro"))
    inertia_variation = (
        float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (inertia_array - mean_inertia) ** 2,
                        axis=(1, 2),
                    )
                )
            )
            / denominator
        )
        if denominator > 1.0e-16
        else math.nan
    )
    return {
        "coupling_ratio": coupling_ratio,
        "normalized_inertia_variation": inertia_variation,
        "coupling_to_required_residual_ratio": coupling_to_residual_ratio,
        "residual_coupling_cosine": residual_coupling_cosine,
        "residual_projection_scale": residual_projection_scale,
    }


def summarize_lambda(
    lambda_scale: float,
    paths: list[Path],
    start_time: float,
    end_margin: float,
    max_horizon: int,
    fixed_end_time: float | None = None,
    min_duration: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    common_end = (
        fixed_end_time
        if fixed_end_time is not None
        else min(last_time(path) for path in paths) - end_margin
    )
    required_duration = max(min_duration, max_horizon * 0.005)
    if common_end - start_time < required_duration:
        raise ValueError(
            f"lambda={lambda_scale:g}: common interval "
            f"{start_time:g}--{common_end:g} s is shorter than "
            f"{required_duration:g} s"
        )
    trials: list[list[dict[str, str]]] = []
    for path in paths:
        trials.extend(read_rows([path], start_time, common_end))

    summary, _ = calculate(trials, max_horizon)
    diagnostic_metrics = coupling_metrics(trials)
    rows: list[dict[str, object]] = []
    for row in summary:
        if row["model"] not in {key for _, key in PREDICTORS}:
            continue
        rows.append(
            {
                "lambda_scale": lambda_scale,
                "evaluation_start_s": start_time,
                "evaluation_end_s": common_end,
                "trajectory_count": len(paths),
                "horizon_steps": row["horizon_steps"],
                "horizon_ms": row["horizon_ms"],
                "predictor": next(
                    label for label, key in PREDICTORS if key == row["model"]
                ),
                "predictor_key": row["model"],
                "samples": row["samples"],
                "omega_rmse_rad_s": row["omega_rmse"],
                "wz_rmse_rad_s": row["wz_rmse"],
                "omega_error_p95_rad_s": row["omega_error_p95"],
                **diagnostic_metrics,
            }
        )
    diagnostic = {
        "lambda_scale": lambda_scale,
        "evaluation_start_s": start_time,
        "evaluation_end_s": common_end,
        "trajectory_count": len(paths),
        **diagnostic_metrics,
    }
    return rows, diagnostic


def plot_results(
    path: Path,
    rows: list[dict[str, object]],
    diagnostics: list[dict[str, object]],
    horizon: int,
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
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.0))
    selected_horizon = [
        row for row in rows if int(row["horizon_steps"]) == horizon
    ]
    for label, key in PREDICTORS:
        selected = sorted(
            (row for row in selected_horizon if row["predictor_key"] == key),
            key=lambda row: float(row["lambda_scale"]),
        )
        lambdas = [float(row["lambda_scale"]) for row in selected]
        axes[0, 0].plot(
            lambdas,
            [float(row["omega_rmse_rad_s"]) for row in selected],
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=1.3,
            markersize=3.8,
            label=label,
        )
        axes[0, 1].plot(
            lambdas,
            [float(row["wz_rmse_rad_s"]) for row in selected],
            color=COLORS[key],
            marker=MARKERS[key],
            linewidth=1.3,
            markersize=3.8,
            label=label,
        )

    diagnostics = sorted(
        diagnostics, key=lambda row: float(row["lambda_scale"])
    )
    lambdas = [float(row["lambda_scale"]) for row in diagnostics]
    axes[1, 0].plot(
        lambdas,
        [100.0 * float(row["coupling_ratio"]) for row in diagnostics],
        color="#1769AA",
        marker="o",
        linewidth=1.3,
        markersize=3.8,
        label=r"$\dot I_G\omega$ / external moment",
    )
    axes[1, 0].plot(
        lambdas,
        [
            100.0 * float(row["normalized_inertia_variation"])
            for row in diagnostics
        ],
        color="#D8902F",
        marker="s",
        linewidth=1.3,
        markersize=3.8,
        label=r"$I_G$ variation",
    )

    by_key = {
        (float(row["lambda_scale"]), str(row["predictor_key"])): row
        for row in selected_horizon
    }
    omega_improvement: list[float] = []
    wz_improvement: list[float] = []
    for lambda_scale in lambdas:
        baseline = by_key[(lambda_scale, "SRBM")]
        full = by_key[(lambda_scale, "IR-linear")]
        omega_base = float(baseline["omega_rmse_rad_s"])
        wz_base = float(baseline["wz_rmse_rad_s"])
        omega_improvement.append(
            100.0
            * (omega_base - float(full["omega_rmse_rad_s"]))
            / omega_base
        )
        wz_improvement.append(
            100.0
            * (wz_base - float(full["wz_rmse_rad_s"]))
            / wz_base
        )
    axes[1, 1].axhline(0.0, color="#777777", linewidth=0.7)
    axes[1, 1].plot(
        lambdas,
        omega_improvement,
        color="#1769AA",
        marker="o",
        linewidth=1.3,
        markersize=3.8,
        label=r"$\omega$ RMSE",
    )
    axes[1, 1].plot(
        lambdas,
        wz_improvement,
        color="#2E8B57",
        marker="s",
        linewidth=1.3,
        markersize=3.8,
        label=r"$\omega_z$ RMSE",
    )

    axes[0, 0].set_ylabel(r"$\omega$ RMSE (rad s$^{-1}$)")
    axes[0, 1].set_ylabel(r"$\omega_z$ RMSE (rad s$^{-1}$)")
    axes[1, 0].set_ylabel("Normalized magnitude (%)")
    axes[1, 1].set_ylabel("Full-model reduction vs. SRBM (%)")
    axes[0, 0].set_title(f"{5 * horizon}-ms angular-velocity prediction")
    axes[0, 1].set_title(f"{5 * horizon}-ms yaw-rate prediction")
    axes[1, 0].set_title("Inertia excitation")
    axes[1, 1].set_title("Prediction-error reduction")
    for axis in axes.flat:
        axis.set_xlabel(r"Leg inertia scale $\lambda$")
        axis.set_xticks(lambdas[::2] if len(lambdas) > 8 else lambdas)
        axis.grid(True, color="#D9D9D9", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    figure.tight_layout(pad=0.8, h_pad=1.1, w_pad=1.0)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--start-time", type=float, default=4.0)
    parser.add_argument("--end-margin", type=float, default=0.25)
    parser.add_argument("--max-horizon", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix for output files, for example 'steady'.",
    )
    parser.add_argument(
        "--per-lambda-window",
        action="store_true",
        help="Use each lambda pair's own common endpoint instead of one global window.",
    )
    args = parser.parse_args()

    trial_rows = read_csv(args.experiment_dir / "trials.csv")
    sources: dict[float, list[Path]] = {}
    for row in trial_rows:
        if int(row["rep"]) != 1 or row["controller"] not in {"srbm", "vicm"}:
            continue
        prediction_path = args.experiment_dir / row["pred_path"]
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        sources.setdefault(float(row["lambda_scale"]), []).append(prediction_path)

    paired_sources = {
        lambda_scale: paths
        for lambda_scale, paths in sources.items()
        if len(paths) == 2
    }
    global_end = (
        None
        if args.per_lambda_window
        else min(
            last_time(path)
            for paths in paired_sources.values()
            for path in paths
        )
        - args.end_margin
    )

    output_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for lambda_scale, paths in sorted(sources.items()):
        if len(paths) != 2:
            skipped.append(
                {
                    "lambda_scale": lambda_scale,
                    "reason": f"expected two trajectories, found {len(paths)}",
                }
            )
            continue
        try:
            rows, diagnostic = summarize_lambda(
                lambda_scale,
                paths,
                args.start_time,
                args.end_margin,
                args.max_horizon,
                global_end,
                args.min_duration,
            )
        except ValueError as error:
            skipped.append({"lambda_scale": lambda_scale, "reason": str(error)})
            continue
        output_rows.extend(rows)
        diagnostics.append(diagnostic)

    if not output_rows:
        raise RuntimeError("no lambda point has a valid paired evaluation interval")
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    summary_path = (
        args.experiment_dir / f"lambda_prediction_consistency{suffix}.csv"
    )
    diagnostic_path = (
        args.experiment_dir / f"lambda_inertia_excitation{suffix}.csv"
    )
    figure_path = (
        args.experiment_dir / f"lambda_prediction_consistency{suffix}.png"
    )
    write_csv(summary_path, output_rows)
    write_csv(diagnostic_path, diagnostics)
    write_csv(
        args.experiment_dir / f"lambda_prediction_skipped{suffix}.csv",
        skipped,
    )
    plot_results(figure_path, output_rows, diagnostics, args.max_horizon)

    print("lambda  H10 SRBM_omega  IG_omega  Full_omega  Full_reduction")
    if global_end is not None:
        print(
            f"Global common evaluation window: "
            f"{args.start_time:.3f}--{global_end:.3f} s"
        )
    for lambda_scale in sorted(float(row["lambda_scale"]) for row in diagnostics):
        selected = {
            str(row["predictor_key"]): row
            for row in output_rows
            if float(row["lambda_scale"]) == lambda_scale
            and int(row["horizon_steps"]) == args.max_horizon
        }
        baseline = float(selected["SRBM"]["omega_rmse_rad_s"])
        full = float(selected["IR-linear"]["omega_rmse_rad_s"])
        print(
            f"{lambda_scale:4.1f}  {baseline:10.6f}  "
            f"{float(selected['VI-frozen']['omega_rmse_rad_s']):8.6f}  "
            f"{full:10.6f}  {100.0 * (baseline - full) / baseline:8.2f}%"
        )
    print(summary_path)
    print(diagnostic_path)
    print(figure_path)
    if skipped:
        print(f"Skipped {len(skipped)} lambda point(s); see lambda_prediction_skipped.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
