#!/usr/bin/env python3
"""Summarize quantized validation-dev evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from frame2kg_eval.metrics.conformity import compute_conformity_from_directory
from frame2kg_eval.metrics.validity import compute_validity_from_directory


@dataclass
class QuantEvalRecord:
    quantization: str
    size: str
    dataset: str
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
    """Extract summary rows keyed by their frame name (e.g. MICRO/MACRO)."""
    summaries: Dict[str, Dict[str, float]] = {}
    with result_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
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


def parse_metadata(csv_path: Path) -> Tuple[str, str, str]:
    """Infer quantization level, model size, and dataset name from the filename."""
    parts = csv_path.stem.split("_")
    if len(parts) < 4 or parts[0] != "quant":
        raise ValueError(f"Unexpected quant evaluation filename: {csv_path.name}")
    quantization = parts[1]
    size = parts[2]
    dataset = "_".join(parts[3:])
    return quantization, size, dataset


def find_prediction_directory(run_root: Path) -> Path:
    """Return the directory containing JSON prediction files for the run."""
    if not run_root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {run_root}")

    candidate_dirs: List[Path] = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        if any(grandchild.suffix == ".json" for grandchild in child.iterdir()):
            candidate_dirs.append(child)
    if not candidate_dirs:
        raise FileNotFoundError(f"No prediction directory with JSON outputs under {run_root}")
    if len(candidate_dirs) > 1:
        raise ValueError(
            f"Multiple prediction directories found under {run_root}: "
            f"{', '.join(dir_path.name for dir_path in candidate_dirs)}"
        )
    return candidate_dirs[0]


def collect_records(results_dir: Path, preds_root: Path) -> List[QuantEvalRecord]:
    csv_files = sorted(results_dir.glob("quant_*_*.csv"))
    if not csv_files:
        raise SystemExit(f"No quant evaluation CSV files found in {results_dir}")

    records: List[QuantEvalRecord] = []
    for csv_path in csv_files:
        quantization, size, dataset = parse_metadata(csv_path)
        pred_run_root = preds_root / f"preds_test_{quantization}_{size.lower()}"
        pred_dir = find_prediction_directory(pred_run_root)

        summaries = parse_summary_rows(csv_path)
        micro = summaries.get("MICRO", {})
        macro = summaries.get("MACRO", {})

        validity_stats = compute_validity_from_directory(pred_dir)
        conformity_stats = compute_conformity_from_directory(pred_dir)

        record = QuantEvalRecord(
            quantization=quantization,
            size=size,
            dataset=dataset,
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
        records.append(record)

    return records


def format_metric(value: float, *, decimals: int) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{decimals}f}"


def write_summary(records: Sequence[QuantEvalRecord], output_path: Path) -> None:
    fieldnames = [
        "quantization",
        "size",
        "dataset",
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
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "quantization": record.quantization,
                    "size": record.size,
                    "dataset": record.dataset,
                    "node_micro_f1": f"{record.node_micro_f1:.6f}" if not math.isnan(record.node_micro_f1) else "",
                    "edge_micro_f1": f"{record.edge_micro_f1:.6f}" if not math.isnan(record.edge_micro_f1) else "",
                    "box_mean_iou": f"{record.box_mean_iou:.6f}" if not math.isnan(record.box_mean_iou) else "",
                    "box_iou@0.5_coverage": f"{record.box_iou_0p5_coverage:.2f}" if not math.isnan(record.box_iou_0p5_coverage) else "",
                    "box_iou@0.75_coverage": f"{record.box_iou_0p75_coverage:.2f}" if not math.isnan(record.box_iou_0p75_coverage) else "",
                    "validity_rate": f"{record.validity_rate:.2f}" if not math.isnan(record.validity_rate) else "",
                    "conformity_rate": f"{record.conformity_rate:.2f}" if not math.isnan(record.conformity_rate) else "",
                    "node_macro_f1": f"{record.node_macro_f1:.6f}" if not math.isnan(record.node_macro_f1) else "",
                    "result_path": record.result_path.as_posix(),
                    "pred_dir": record.pred_dir.as_posix(),
                }
            )


def print_summary(records: Sequence[QuantEvalRecord]) -> None:
    print(
        "Quantized eval metrics (node_micro_f1, edge_micro_f1, box_mean_iou, "
        "box_iou@0.5_coverage, box_iou@0.75_coverage, validity_rate, conformity_rate, "
        "node_macro_f1):"
    )
    for record in records:
        metrics = (
            format_metric(record.node_micro_f1, decimals=4),
            format_metric(record.edge_micro_f1, decimals=4),
            format_metric(record.box_mean_iou, decimals=4),
            format_metric(record.box_iou_0p5_coverage, decimals=2),
            format_metric(record.box_iou_0p75_coverage, decimals=2),
            format_metric(record.validity_rate, decimals=2),
            format_metric(record.conformity_rate, decimals=2),
            format_metric(record.node_macro_f1, decimals=4),
        )
        metric_str = ", ".join(metrics)
        print(f"  {record.size}/{record.quantization} [{record.dataset}]: {metric_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize quantized val-dev evaluation outputs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("quantEval"),
        help="Directory containing quant evaluation CSV files",
    )
    parser.add_argument(
        "--preds-root",
        type=Path,
        default=Path("quantEval"),
        help="Root directory containing quantized prediction outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for the output summary CSV (default: results_dir/quant_summary.csv)",
    )
    args = parser.parse_args()

    records = collect_records(args.results_dir, args.preds_root)
    records.sort(key=lambda r: (r.size, r.quantization))

    output_path = args.out or (args.results_dir / "quant_summary.csv")
    write_summary(records, output_path)

    print_summary(records)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
