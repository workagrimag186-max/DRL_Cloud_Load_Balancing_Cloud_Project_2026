"""
Tests for Issue #28 — Experiment E4: 100x Flash-Sale Burst.
Validates that:
1. All 6 agents (PPO, DQN, RR, WRR, LC, Threshold) are evaluated on E4 100x burst.
2. 5 repetitions with test seeds 142-146 are executed.
3. 5 result files per agent exist and are not empty (30 total files).
4. Results compliance and upper bound / infrastructure ceiling verification.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
from scipy import stats

from evaluation.run_e4_100x import (
    AGENT_NAMES,
    AGENT_ALIASES,
    E4_SCENARIO,
    E4_SEEDS,
    run_e4_experiments,
)


def test_e4_100x_runner_quick(tmp_path):
    """Test runner with truncated steps to verify end-to-end pipeline execution."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ppo_path = root_dir / "models/ppo/best_model.zip"
    dqn_path = root_dir / "models/dqn/best_model.zip"

    summaries, hypothesis_result = run_e4_experiments(
        output_dir=tmp_path,
        seeds=(142,),
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        max_steps=50,
    )

    assert len(summaries) == 6
    for s in summaries:
        assert s["scenario"] == E4_SCENARIO
        assert s["seed"] == 142
        assert s["p95_latency_ms"] > 0
        assert s["throughput_rps"] > 0

        # Check file exists and has rows
        file_path = tmp_path / f"e4_100x_{s['agent']}_seed142.csv"
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_e4_100x_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e4_100x/ and infrastructure ceiling report."""
    results_dir = Path("experiments/results/e4_100x")
    summary_file = results_dir / "e4_100x_summary.csv"
    report_file = results_dir / "hypothesis_test_report.json"

    if not summary_file.exists():
        pytest.skip("Full E4 experiment has not been executed yet.")

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 30, f"Expected 30 summary records (6 agents x 5 seeds), got {len(reader)}"

    agents_found = {row["agent"] for row in reader}
    expected_agents = set(AGENT_NAMES.keys())
    assert agents_found == expected_agents, f"Missing agents: {expected_agents - agents_found}"

    seeds_found = {int(row["seed"]) for row in reader}
    assert seeds_found == set(E4_SEEDS), f"Missing seeds: {set(E4_SEEDS) - seeds_found}"

    # Verify individual run files exist
    for row in reader:
        agent = row["agent"]
        seed = row["seed"]
        agent_file = results_dir / f"e4_100x_{agent}_seed{seed}.csv"
        assert agent_file.exists(), f"Missing result file: {agent_file}"
        assert agent_file.stat().st_size > 0, f"Empty result file: {agent_file}"

    # Verify report
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert "ppo_mean_p95_ms" in report
    assert "rr_mean_p95_ms" in report
    assert report["ppo_mean_p95_ms"] < report["rr_mean_p95_ms"]
