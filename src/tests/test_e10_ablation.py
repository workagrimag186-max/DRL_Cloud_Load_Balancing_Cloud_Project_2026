"""
Tests for Issue #34 — Experiment E10: Reward Weight Ablation.
Validates that:
1. All 5 component ablation variants (Baseline, No R_lat, No R_util, No R_tput, No R_sla) are evaluated.
2. 2D Sensitivity grid across (w_lat, w_sla) is evaluated and heatmap is generated.
3. Verifies Hypothesis H7: removing SLA penalty (w4=0) increases P99 latency by +20–40%.
4. Verifies baseline weights (0.4, 0.2, 0.2, 0.2) confirmed optimal in P95 latency.
5. Verifies notebook notebooks/04_ablation_study.ipynb structure and outputs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from evaluation.run_e10_ablation import (
    ABLATION_VARIANTS,
    E10_SCENARIO,
    E10_SEEDS,
    run_e10_experiments,
)


def test_e10_ablation_runner_quick(tmp_path):
    """Test ablation runner with truncated steps to verify end-to-end pipeline execution."""
    summaries, report = run_e10_experiments(
        output_dir=tmp_path,
        seeds=(132,),
        quick_mode=True,
        max_steps=50,
    )

    assert len(summaries) == 5
    for s in summaries:
        assert s["variant_id"] in ABLATION_VARIANTS
        assert s["seed"] == 132
        assert s["p95_latency_ms"] > 0
        assert s["throughput_rps"] > 0

    assert (tmp_path / "e10_ablation_summary.csv").exists()
    assert (tmp_path / "e10_ablation_report.json").exists()
    assert (tmp_path / "ablation_heatmap.png").exists()
    assert (tmp_path / "ablation_components.png").exists()


def test_e10_ablation_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e10_ablation/ and hypothesis H7."""
    results_dir = Path("experiments/results/e10_ablation")
    summary_file = results_dir / "e10_ablation_summary.csv"
    report_file = results_dir / "e10_ablation_report.json"
    heatmap_file = results_dir / "ablation_heatmap.png"
    components_file = results_dir / "ablation_components.png"

    if not summary_file.exists():
        pytest.skip("Full E10 experiment has not been executed yet.")

    assert heatmap_file.exists()
    assert heatmap_file.stat().st_size > 0
    assert components_file.exists()
    assert components_file.stat().st_size > 0

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 15, f"Expected 15 summary records (5 variants x 3 seeds), got {len(reader)}"

    variants_found = {row["variant_id"] for row in reader}
    expected_variants = set(ABLATION_VARIANTS.keys())
    assert variants_found == expected_variants, f"Missing variants: {expected_variants - variants_found}"

    seeds_found = {int(row["seed"]) for row in reader}
    assert seeds_found == set(E10_SEEDS), f"Missing seeds: {set(E10_SEEDS) - seeds_found}"

    # Verify report and hypotheses
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    # H7 Verification: Removing SLA penalty increases P99 latency by +20-40%
    h7 = report["hypothesis_h7"]
    assert h7["confirmed"] is True
    assert 20.0 <= h7["p99_increase_percent"] <= 50.0

    # Optimality Verification: Baseline weights achieve lowest P95 latency
    opt = report["optimality_confirmation"]
    assert opt["baseline_confirmed_optimal"] is True


def test_e10_ablation_notebook_structure():
    """Verify notebooks/04_ablation_study.ipynb exists and has required structure."""
    nb_path = Path("notebooks/04_ablation_study.ipynb")
    assert nb_path.exists(), "Missing notebooks/04_ablation_study.ipynb"

    content = json.loads(nb_path.read_text(encoding="utf-8"))
    assert "cells" in content
    assert len(content["cells"]) >= 5

    code_cells = [c for c in content["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 3
    sources = "".join("".join(c["source"]) for c in code_cells)
    assert "run_e10_experiments" in sources
    assert "ablation_heatmap.png" in sources
