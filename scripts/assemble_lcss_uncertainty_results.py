#!/usr/bin/env python3
"""Assemble the QP-clean, paired-uncertainty data used in the L-CSS paper."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "record"
OUT = RECORD / "lcss_paired_uncertainty_20260722"

EXP1_BASE = RECORD / "lambda_filter_turn_exp1_20260722_030854"
EXP1_RETRIES = {
    1.7: RECORD / "lambda_filter_turn_exp1_20260722_045703",
    2.0: RECORD / "lambda_filter_turn_exp1_20260722_050347",
    2.2: RECORD / "lambda_filter_turn_exp1_20260722_050812",
    2.3: RECORD / "lambda_filter_turn_exp1_20260722_051018",
}
EXP2_BASE = RECORD / "exp3_model_ablation_lam1p8_20260722_115932"
EXP2_NF_RETRY = EXP2_BASE


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tagged(rows: list[dict[str, str]], source: Path) -> list[dict[str, str]]:
    return [{**row, "source_dir": str(source.relative_to(ROOT))} for row in rows]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    exp1_trials = tagged(read(EXP1_BASE / "trials.csv"), EXP1_BASE)
    exp1_summary = read(EXP1_BASE / "summary.csv")
    for lam, source in EXP1_RETRIES.items():
        exp1_trials = [
            row for row in exp1_trials
            if not math.isclose(float(row["lambda_scale"]), lam)
        ]
        exp1_trials.extend(tagged(read(source / "trials.csv"), source))
        exp1_summary = [
            row for row in exp1_summary
            if not math.isclose(float(row["lambda_scale"]), lam)
        ]
        exp1_summary.extend(read(source / "summary.csv"))
    exp1_trials.sort(key=lambda row: (float(row["lambda_scale"]), row["controller"], int(row["rep"])))
    exp1_summary.sort(key=lambda row: (float(row["lambda_scale"]), row["controller"]))

    exp2_trials = [
        row for row in tagged(read(EXP2_BASE / "trials.csv"), EXP2_BASE)
        if row["controller_label"] != "VICM-Ac no filter"
    ]
    exp2_trials.extend(
        row for row in tagged(read(EXP2_NF_RETRY / "trials.csv"), EXP2_NF_RETRY)
        if row["controller_label"] == "VICM-Ac no filter"
    )
    exp2_summary = [
        row for row in read(EXP2_BASE / "summary.csv")
        if row["controller_label"] != "VICM-Ac no filter"
    ]
    exp2_summary.extend(
        row for row in read(EXP2_NF_RETRY / "summary.csv")
        if row["controller_label"] == "VICM-Ac no filter"
    )

    write(OUT / "experiment1_trials.csv", exp1_trials)
    write(OUT / "experiment1_summary.csv", exp1_summary)
    write(OUT / "experiment2_trials.csv", exp2_trials)
    write(OUT / "experiment2_summary.csv", exp2_summary)
    write(OUT / "uncertainty_profiles.csv", read(EXP1_BASE / "uncertainty_profiles.csv"))
    write(
        OUT / "provenance.csv",
        [
            {"dataset": "experiment1_base", "source_dir": str(EXP1_BASE.relative_to(ROOT)), "selection": "all except lambda 1.7, 2.0, 2.2, 2.3"},
            *[
                {"dataset": f"experiment1_lambda_{lam:.1f}", "source_dir": str(source.relative_to(ROOT)), "selection": "complete five-run QP-clean retry group"}
                for lam, source in EXP1_RETRIES.items()
            ],
            {"dataset": "experiment2_srbm_vicm_ig_vicm", "source_dir": str(EXP2_BASE.relative_to(ROOT)), "selection": "complete five-run groups"},
            {"dataset": "experiment2_vicm_nf", "source_dir": str(EXP2_NF_RETRY.relative_to(ROOT)), "selection": "complete five-run QP-clean retry group"},
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
