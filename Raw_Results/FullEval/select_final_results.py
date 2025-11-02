#!/usr/bin/env python3
"""Aggregate final full-dataset evaluation metrics for release checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from frame2kg_eval.metrics.conformity import compute_conformity_from_directory
from frame2kg_eval.metrics.validity import compute_validity_from_directory


@dataclass(frozen=True)
class FinalRunDef:
    """Static configuration describing a final-eval checkpoint."""

    slug: str
    size: str
    variant: str
    ckpt_label: str
    pred_subpath: Path


FINAL_RUNS: Sequence[FinalRunDef] = (
    FinalRunDef(
        slug="3B-QKVO_final",
        size="3B",
        variant="QKVO",
        ckpt_label="final",
        pred_subpath=Path("3B-QKVO") / "4",
    ),
    FinalRunDef(
        slug="3B-QKVO-Gate_final",
        size="3B",
        variant="QKVO-Gate",
        ckpt_label="final",
        pred_subpath=Path("3B-QKVO-Gate") / "4",
    ),
    FinalRunDef(
        slug="7B-QKVO_best",
        size="7B",
        variant="QKVO",
        ckpt_label="best",
        pred_subpath=Path("7B-QKVO") / "3",
    ),
    FinalRunDef(
        slug="7B-QKVO-Gate_step1k",
        size="7B",
        variant="QKVO-Gate",
        ckpt_label="step1k",
        pred_subpath=Path("7B-QKVO-Gate") / "1",
    ),
)


@dataclass
class FinalEvalRecord:
    run: FinalRunDef
    result_path: Path
    pred_dir: Path
    node_micro_f1: float
    edge_micro_f1: float
    box_mean_iou: float
    box_iou_0p5_coverage: float
    box_iou_0p75_coverage: float
    validity_rate: float
    conformity_rate: float
    node_macro_f1: float
    extra: Dict[str, float] = field(default_factory=dict)


def parse_summary_rows(result_path: Path) -> Dict[str, Dict[str, float]]:
    """Extract the summary metrics rows from a result CSV file."""

    summaries: Dict[str, Dict[str, float]] = {}
    with result_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if row.get("video_id") != "SUMMARY":
                continue
            frame_no = row.get("frame_no", "").strip().upper()
            metrics: Dict[str, float] = {}
            for key, value in row.items():
                if key in {"video_id", "frame_no"}:
                    continue
                if value in (None, ""):
                    continue
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            summaries[frame_no] = metrics
    return summaries


def collect_records(results_dir: Path, preds_dir: Path, run_defs: Sequence[FinalRunDef]) -> List[FinalEvalRecord]:
    records: List[FinalEvalRecord] = []

    for run in run_defs:
        csv_path = results_dir / f"{run.slug}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing evaluation output: {csv_path}")

        pred_dir = preds_dir / run.pred_subpath
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction directory for {run.slug}: {pred_dir}")

        summaries = parse_summary_rows(csv_path)
        micro = summaries.get("MICRO", {})
        macro = summaries.get("MACRO", {})

        validity_stats = compute_validity_from_directory(pred_dir)
        conformity_stats = compute_conformity_from_directory(pred_dir)

        records.append(
            FinalEvalRecord(
                run=run,
                result_path=csv_path,
                pred_dir=pred_dir,
                node_micro_f1=micro.get("node_f1", float("nan")),
                edge_micro_f1=micro.get("edge_f1", float("nan")),
                box_mean_iou=micro.get("box_mean_iou", float("nan")),
                box_iou_0p5_coverage=micro.get("box_iou@0.5_coverage", float("nan")),
                box_iou_0p75_coverage=micro.get("box_iou@0.75_coverage", float("nan")),
                validity_rate=validity_stats.get("validity_rate", float("nan")),
                conformity_rate=conformity_stats.get("conformity_rate_total", float("nan")),
                node_macro_f1=macro.get("node_f1", float("nan")),
                extra={
                    "edge_macro_f1": macro.get("edge_f1", float("nan")),
                    "box_macro_mean_iou": macro.get("box_mean_iou", float("nan")),
                    "box_macro_iou@0.5_coverage": macro.get("box_iou@0.5_coverage", float("nan")),
                    "box_macro_iou@0.75_coverage": macro.get("box_iou@0.75_coverage", float("nan")),
                },
            )
        )

    return records


def format_metric(value: float, *, decimals: int) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def write_summary(records: Iterable[FinalEvalRecord], output_path: Path) -> None:
    fieldnames = [
        "size",
        "variant",
        "checkpoint",
        "node_micro_f1",
        "edge_micro_f1",
        "box_mean_iou",
        "box_iou@0.5_coverage",
        "box_iou@0.75_coverage",
        "validity_rate",
        "conformity_rate",
        "node_macro_f1",
        "result_path",
        "pred_dir",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "size": rec.run.size,
                    "variant": rec.run.variant,
                    "checkpoint": rec.run.ckpt_label,
                    "node_micro_f1": f"{rec.node_micro_f1:.6f}" if not math.isnan(rec.node_micro_f1) else "",
                    "edge_micro_f1": f"{rec.edge_micro_f1:.6f}" if not math.isnan(rec.edge_micro_f1) else "",
                    "box_mean_iou": f"{rec.box_mean_iou:.6f}" if not math.isnan(rec.box_mean_iou) else "",
                    "box_iou@0.5_coverage": f"{rec.box_iou_0p5_coverage:.2f}" if not math.isnan(rec.box_iou_0p5_coverage) else "",
                    "box_iou@0.75_coverage": f"{rec.box_iou_0p75_coverage:.2f}" if not math.isnan(rec.box_iou_0p75_coverage) else "",
                    "validity_rate": f"{rec.validity_rate:.2f}" if not math.isnan(rec.validity_rate) else "",
                    "conformity_rate": f"{rec.conformity_rate:.2f}" if not math.isnan(rec.conformity_rate) else "",
                    "node_macro_f1": f"{rec.node_macro_f1:.6f}" if not math.isnan(rec.node_macro_f1) else "",
                    "result_path": rec.result_path.as_posix(),
                    "pred_dir": rec.pred_dir.as_posix(),
                }
            )


def print_summary(records: Sequence[FinalEvalRecord]) -> None:
    print(
        "Final testing metrics (node_micro_f1, edge_micro_f1, box_mean_iou, "
        "box_iou@0.5_coverage, box_iou@0.75_coverage, validity_rate, conformity_rate, "
        "node_macro_f1):"
    )
    for rec in records:
        metrics = (
            format_metric(rec.node_micro_f1, decimals=4),
            format_metric(rec.edge_micro_f1, decimals=4),
            format_metric(rec.box_mean_iou, decimals=4),
            format_metric(rec.box_iou_0p5_coverage, decimals=2),
            format_metric(rec.box_iou_0p75_coverage, decimals=2),
            format_metric(rec.validity_rate, decimals=2),
            format_metric(rec.conformity_rate, decimals=2),
            format_metric(rec.node_macro_f1, decimals=4),
        )
        metric_str = ", ".join(metrics)
        print(f"  {rec.run.size}/{rec.run.variant}: {rec.run.ckpt_label} -> {metric_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize final full testing evaluation outputs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_final"),
        help="Directory containing final evaluation CSV files",
    )
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("preds_full"),
        help="Root directory containing prediction folders",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional subset of configured run slugs to include",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for the summary CSV (default: results_dir/final_summary.csv)",
    )
    args = parser.parse_args()

    run_map = {run.slug: run for run in FINAL_RUNS}
    if args.runs:
        unknown = [slug for slug in args.runs if slug not in run_map]
        if unknown:
            raise SystemExit(f"Unknown run slug(s): {', '.join(unknown)}")
        selected_runs = [run_map[slug] for slug in args.runs]
    else:
        selected_runs = list(FINAL_RUNS)

    records = collect_records(args.results_dir, args.preds_dir, selected_runs)
    records.sort(key=lambda r: (r.run.size, r.run.variant, r.run.ckpt_label))

    output_path = args.out or (args.results_dir / "final_summary.csv")
    write_summary(records, output_path)

    print_summary(records)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
