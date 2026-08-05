#!/usr/bin/env python3
"""Assemble the final Experiment 2 metrics from matched retained trials."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "record" / "exp2_discrete_momentum_ablation_lam1p8_20260729_144556"
DM_DIR = ROOT / "record" / "exp2_discrete_momentum_ablation_lam1p8_20260802_193258"
IR_DIR = ROOT / "record" / "exp3_model_ablation_lam1p8_20260729_145758"
TIMING_DIR = ROOT / "record" / "exp2_final_ablation_lam1p8_20260802_180954"
OUT_DIR = ROOT / "record" / "exp2_final_ablation_lam1p8_assembled_20260802"

SOURCES = (
    ("srbm", "SRBM", BASE_DIR, "srbm"),
    ("vicm_ig", "VI-CMPC", BASE_DIR, "vicm_ig"),
    ("ir_cmpc", "IR-CMPC", IR_DIR, "vicm_ac"),
    ("dm_preview", "DM-CMPC", DM_DIR, "dm_preview"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def h10_rmse(path: Path) -> float:
    errors: list[float] = []
    for row in read_rows(path):
        if int(row["horizon_steps"]) == 10 and float(row["origin_time"]) >= 4.0:
            errors.append(float(row["err_wz"]))
    if not errors:
        raise RuntimeError(f"No H10 errors in {path}")
    return math.sqrt(statistics.mean(error * error for error in errors))


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> int:
    timing = {row["controller"]: row for row in read_rows(TIMING_DIR / "trials.csv")}
    trials: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []

    for controller, label, source_dir, source_controller in SOURCES:
        source_trials = [
            row
            for row in read_rows(source_dir / "trials.csv")
            if row["controller"] == source_controller
        ]
        if len(source_trials) != 3:
            raise RuntimeError(f"Expected three {label} trials in {source_dir}")
        survival: list[float] = []
        h10: list[float] = []
        for row in source_trials:
            rep = int(row["rep"])
            horizon = next(source_dir.glob(f"*_{source_controller}_r{rep}_mpc_horizon.csv"))
            survival_value = float(row["final_time"])
            h10_value = h10_rmse(horizon)
            survival.append(survival_value)
            h10.append(h10_value)
            trials.append(
                {
                    "controller": controller,
                    "controller_label": label,
                    "rep": rep,
                    "final_time": survival_value,
                    "completed_30s": int(survival_value >= 29.999),
                    "h10_wz_prediction_rmse": h10_value,
                    "source_dir": source_dir.relative_to(ROOT),
                    "source_horizon": horizon.name,
                }
            )

        timing_row = timing[controller]
        summary.append(
            {
                "controller": controller,
                "controller_label": label,
                "n": len(survival),
                "completed_30s": sum(value >= 29.999 for value in survival),
                "mean_survival_time": statistics.mean(survival),
                "sample_sd_survival_time": sample_sd(survival),
                "mean_h10_wz_prediction_rmse": statistics.mean(h10),
                "sample_sd_h10_wz_prediction_rmse": sample_sd(h10),
                "mean_mpc_wall_ms": float(timing_row["mpc_avg_wall_ms"]),
                "max_mpc_wall_ms": float(timing_row["mpc_max_wall_ms"]),
                "mpc_timing_samples": int(timing_row["mpc_samples"]),
            }
        )
        provenance.append(
            {
                "controller": controller,
                "closed_loop_source": source_dir.relative_to(ROOT),
                "timing_source": TIMING_DIR.relative_to(ROOT),
                "selection": "matched deterministic lambda=1.8 trials",
            }
        )

    write_rows(OUT_DIR / "trials.csv", trials)
    write_rows(OUT_DIR / "summary.csv", summary)
    write_rows(OUT_DIR / "provenance.csv", provenance)
    print(OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
