#!/usr/bin/env python3
"""Merge a corrected SRBM push map with an existing IR-CMPC push map."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from run_exp4_push_recovery_map import TRIAL_FIELDS, plot_recovery
from vicm_experiment_lib import RECORD_DIR, read_rows, write_csv


def selected_rows(source_dir: Path, controller: str) -> list[dict[str, object]]:
    rows = [
        dict(row)
        for row in read_rows(source_dir / "trials.csv")
        if row["controller"] == controller
    ]
    if not rows:
        raise RuntimeError(f"no {controller} rows in {source_dir}")
    for row in rows:
        for field in ("trace_path", "log_path"):
            if row.get(field):
                row[field] = str((source_dir / str(row[field])).resolve())
    return rows


def grid_keys(rows: list[dict[str, object]]) -> set[tuple[float, float, int]]:
    return {
        (
            float(row["push_angle_deg"]),
            float(row["push_force"]),
            int(row["rep"]),
        )
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("srbm_dir", type=Path)
    parser.add_argument("ir_dir", type=Path)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    srbm_rows = selected_rows(args.srbm_dir.resolve(), "srbm")
    ir_rows = selected_rows(args.ir_dir.resolve(), "vicm_ac")
    if grid_keys(srbm_rows) != grid_keys(ir_rows):
        missing_ir = sorted(grid_keys(srbm_rows) - grid_keys(ir_rows))
        missing_srbm = sorted(grid_keys(ir_rows) - grid_keys(srbm_rows))
        raise RuntimeError(
            "push grids do not match: "
            f"missing IR-CMPC={missing_ir[:5]}, missing SRBM={missing_srbm[:5]}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else RECORD_DIR / f"exp4_push_recovery_corrected_srbm_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = srbm_rows + ir_rows
    with (out_dir / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRIAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_csv(
        out_dir / "metadata.csv",
        [
            {
                "srbm_source": str(args.srbm_dir.resolve()),
                "ir_cmpc_source": str(args.ir_dir.resolve()),
                "srbm_trials": len(srbm_rows),
                "ir_cmpc_trials": len(ir_rows),
                "boundary_threshold": args.boundary_threshold,
            }
        ],
    )
    plot_recovery(out_dir, rows, args.boundary_threshold)
    print(f"OUT={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
