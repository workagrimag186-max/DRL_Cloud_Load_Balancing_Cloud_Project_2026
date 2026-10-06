"""
Unit and Integration Tests for Statistical Analysis Pipeline.
Phase 7: Results & Statistical Validation - Issue #35
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = ROOT / "experiments" / "analysis"


def test_summary_statistics_file_exists_and_valid():
    """Verify summary_statistics.csv existence, schema, and sample sizes."""
    summary_csv = ANALYSIS_DIR / "summary_statistics.csv"
    assert summary_csv.exists(), f"Missing file: {summary_csv}"

    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 0, "summary_statistics.csv is empty"
    required_cols = {"scenario", "agent", "metric", "n", "mean", "ci_95", "mean_ci_str"}
    assert required_cols.issubset(set(reader[0].keys())), f"Missing columns in summary_statistics.csv: {reader[0].keys()}"

    for row in reader:
        n = int(row["n"])
        assert n >= 5, f"Expected n >= 5 reps per cell, found {n} for {row['scenario']} - {row['agent']} - {row['metric']}"
        mean_val = float(row["mean"])
        ci_val = float(row["ci_95"])
        assert ci_val >= 0, f"Confidence interval half-width must be non-negative: {ci_val}"


def test_statistical_tests_structure_and_p_values():
    """Verify statistical_tests.csv schema, non-null p-values, and >= 5 pairs."""
    stat_csv = ANALYSIS_DIR / "statistical_tests.csv"
    assert stat_csv.exists(), f"Missing file: {stat_csv}"

    with open(stat_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) > 0, "statistical_tests.csv is empty"
    required_cols = {
        "scenario", "metric", "comparison", "n_pairs",
        "t_stat", "p_value_ttest", "w_stat", "p_value_wilcoxon", "significant_ttest_05"
    }
    assert required_cols.issubset(set(reader[0].keys())), f"Missing columns in statistical_tests.csv: {reader[0].keys()}"

    for row in reader:
        n_pairs = int(row["n_pairs"])
        assert n_pairs >= 5, f"Expected n_pairs >= 5, found {n_pairs} for {row['comparison']} {row['metric']}"
        assert row["p_value_ttest"] != "", f"Empty p_value_ttest in {row['comparison']} {row['metric']}"
        p_val = float(row["p_value_ttest"])
        assert 0.0 <= p_val <= 1.0, f"p_value_ttest out of range [0, 1]: {p_val}"


def test_primary_claim_ppo_vs_rr_on_e2_significance():
    """
    Acceptance Criteria:
    statistical_tests.csv shows p < 0.05 for PPO vs RR on E2 (primary claim).
    """
    stat_csv = ANALYSIS_DIR / "statistical_tests.csv"
    with open(stat_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Find PPO vs RR on E2 for p95_latency_ms
    matches = [
        r for r in reader
        if r["scenario"] == "e2_10x"
        and r["comparison"] == "PPO vs RR"
        and r["metric"] == "p95_latency_ms"
    ]
    assert len(matches) == 1, f"Expected 1 record for PPO vs RR p95_latency_ms on e2_10x, found {len(matches)}"

    row = matches[0]
    p_val = float(row["p_value_ttest"])
    assert p_val < 0.05, f"Primary claim failed: p-value {p_val} is not < 0.05"
    assert row["significant_ttest_05"] == "True"


def test_statistical_analysis_pipeline_execution():
    """Verify that scripts/statistical_analysis.py runs cleanly."""
    import scripts.statistical_analysis as sa
    sa.main()

    assert (ANALYSIS_DIR / "summary_statistics.csv").exists()
    assert (ANALYSIS_DIR / "statistical_tests.csv").exists()
