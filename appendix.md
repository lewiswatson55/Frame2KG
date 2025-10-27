# Appendix: Additional Tables for *Frame2KG–YC2*

> **Scope & numbering.** This appendix is maintained in-repo while the paper is under review (no appendix allowed). Table numbers here use an **A.x** scheme and may not match the final paper’s numbering.

**Contents**
- [A1. Validation\_dev winners and selection details](#a1-validation_dev-winners-and-selection-details)
- [A2. Ablation: effect of adding MLP (GateProj/Up/Down)](#a2-ablation-effect-of-adding-mlp-gateprojupdown)
- [A3. Dataset statistics (training split)](#a3-dataset-statistics-training-split)
- [A4. Frozen frame2kg-eval default config](#a4-frozen-frame2kg-eval-default-config)
- [A5. Matcher sensitivity to IoU gate and blend weight](#a5-matcher-sensitivity-to-iou-gate-and-blend-weight)
- [A6. Composite-node diagnostic](#a6-composite-node-diagnostic)
- [A7. Quantisation impact](#a7-quantisation-impact)
- [Notation & conventions](#notation--conventions)

---

## A1. Validation_dev winners and selection details

| Model            | Checkpoint | Node F1_μ | Edge F1_μ | Box IoU | IoU@0.5 | IoU@0.75 | Validity (%) | Conform. (%) | Node F1_M |
|------------------|------------|-----------|-----------|---------|---------|----------|--------------|--------------|-----------|
| 3B/QKVO          | final      | 0.636     | 0.204     | 0.603   | 0.650   | 0.270    | 99           | 99           | 0.653     |
| 3B/QKVO_Gate     | final      | 0.658     | 0.236     | 0.613   | 0.690   | 0.250    | 100          | 100          | 0.667     |
| 7B/QKVO          | best       | 0.669     | 0.257     | 0.619   | 0.680   | 0.310    | 100          | 99           | 0.664     |
| 7B/QKVO_Gate     | step1k     | 0.664     | **0.290** | **0.620** | **0.720** | 0.280    | 99           | 99           | **0.680** |

**Notes.** Validation\_dev winners per family with key metrics (rounded). The full per-checkpoint CSV and prediction directories are archived with the release for reproducibility.

---

## A2. Ablation: effect of adding MLP (GateProj/Up/Down)

| Model         | Node F1_μ | Edge F1_μ | IoU mean |
|---------------|-----------|-----------|----------|
| 3B--QKVO      | 0.597     | 0.187     | 0.600    |
| 3B--QKVO-Gate | 0.616     | **0.208** | **0.609**|
| 7B--QKVO      | 0.607     | 0.195     | 0.601    |
| 7B--QKVO-Gate | **0.621** | 0.204     | 0.605    |

**Notes.** Gate improves predicates and localisation; 7B helps nodes.

---

## A3. Dataset statistics (training split)

| Label          | Count | Predicate       | Count |
|----------------|------:|-----------------|------:|
| countertop     |  2753 | `on`            | 34046 |
| cutting board  |  1919 | `in`            |  7724 |
| hand           |  1898 | `next_to`       |  7665 |
| person         |  1654 | `holding`       |  7211 |
| man            |  1566 | `in_front_of`   |  5571 |
| stove          |  1344 | `left_of`       |  3242 |
| plate          |  1340 | `behind`        |  3005 |
| woman          |  1332 | `supports`      |  2788 |
| bowl           |  1265 | `includes`      |  2218 |
| frying pan     |  1086 | `right_of`      |  1951 |

**Notes.** Most frequent node labels and predicates in the *training* split.

---

## A4. Frozen frame2kg-eval default config

| **Setting**                 | **Default**                 | **Notes**                                 |
|----------------------------|-----------------------------|-------------------------------------------|
| Node IoU gate τ            | 0.3                         | Stage-1 box gating                        |
| Blend weight α             | 0.7                         | IoU vs. text in Stage-2                   |
| Text mode                  | `semantic`                  | Options: `tfidf`, `semantic`, `hybrid`    |
| Text fields                | `labels, attributes`        | Used to form node text                    |
| Text floor                 | 0.25                        | Min cosine sim to count                   |
| Sentence encoder           | `all-MiniLM-L6-v2`          | `sentence-transformers/` model            |
| Predicate mode             | `normalised`                | String eq. after normalisation            |
| Semantic predicate θ       | 0.6                         | Not used                                  |
| Include invalid            | `true`                      | Invalid JSON counted (empties too)        |
| Strict mode                | `false`                     | If true, penalises invalids as FPs        |
| Aggregation                | `micro, macro`              | Both are reported                         |
| Dataset                    | `REDACTED/Frame2KG-YC2`     | HF dataset name                           |
| Default split              | `validation_dev`            | For selection/sweeps                      |
| Output format              | `csv`                       | Deterministic summaries + per-frame       |
| Verbose                    | `true`                      | Detailed logs enabled                     |
| Sweep τ                    | `[0.3, 0.5, 0.7]`           | Matcher sensitivity                        |
| Sweep α                    | `[0.5, 0.7, 0.85]`          | Matcher sensitivity                        |

**Notes.** Frozen `frame2kg-eval` config used in the paper’s experiments.

---

## A5. Matcher sensitivity to IoU gate and blend weight

| Model                     | τ/α       | Node F1_μ | Edge F1_μ | Comb. F1 | n   |
|---------------------------|-----------|-----------|-----------|----------|-----|
| 3B--QKVO-Gate (final)     | **0.3/0.7** | **0.650** | **0.238** | **0.444** | 100 |
|                           | 0.3/0.6   | 0.650     | 0.236     | 0.443    | 100 |
| 7B--QKVO-Gate (step1k)    | **0.3/0.7** | **0.664** | **0.294** | **0.479** |  99 |
|                           | 0.3/0.6   | 0.664     | 0.294     | 0.479    |  99 |

**Notes.** τ is the IoU gate for stage-1 node alignment; α is the blend weight between localisation and text similarity in stage-2. All other eval settings frozen.

---

## A6. Composite-node diagnostic

| Model                     | Split (pred→GT) FN | Merge (GT→pred) FP |
|---------------------------|--------------------|--------------------|
| 3B--QKVO-Gate (final)     | 2.7% (6/221)       | 5.5% (10/181)      |
| 7B--QKVO-Gate (step1k)    | 1.9% (4/212)       | 2.1% (4/188)       |

**Notes.** Splits: multiple predicted nodes align to a single GT node (as % of node FNs). Merges: multiple GT nodes align to a single predicted node (as % of node FPs). Evaluated on *val_dev* (100 frames).

---

## A7. Quantisation impact

| Model          | Format | Node F1_μ        | Edge F1_μ        | IoU mean | s/graph (↓) \[× vs FP16] |
|----------------|--------|------------------|------------------|----------|---------------------------|
| 3B--QKVO-Gate  | FP16   | 0.650            | 0.238            | 0.613    | 30.3 \[1.00×]            |
|                | INT8   | 0.646 (Δ=−0.012) | 0.254 (Δ=+0.018) | 0.619    | 101.9 \[0.30×]           |
|                | INT4   | 0.636 (Δ=−0.022) | 0.232 (Δ=−0.004) | 0.612    |  43.7 \[0.69×]           |
| 7B--QKVO-Gate  | FP16   | 0.664            | 0.294            | 0.620    | 27.8 \[1.00×]            |
|                | INT8   | 0.686 (Δ=+0.022) | 0.287 (Δ=−0.003) | 0.620    |  95.3 \[0.29×]           |
|                | INT4   | 0.658 (Δ=−0.006) | 0.278 (Δ=−0.012) | 0.616    |  41.4 \[0.67×]           |

**Notes.** Post-training quantisation on *val_dev* (same eval defaults: τ=0.3, α=0.7, semantic). Δ values are absolute changes vs the FP16 run for the same model and split. Speed is mean end-to-end wall time per frame from `manifest.csv`; bracket shows throughput relative to FP16.
---

## Notation & conventions

- **F1\_μ** / **F1\_M** denote micro / macro F1.  
- **τ** = IoU gate (stage-1 node alignment). **α** = blend weight between IoU and text similarity (stage-2).  
- **Δ** entries show absolute change vs the FP16 baseline for the same model and split.  
- All metrics are computed with the frozen `frame2kg-eval` configuration in Table A4 unless noted.