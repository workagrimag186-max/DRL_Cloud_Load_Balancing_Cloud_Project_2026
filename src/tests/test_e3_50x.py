"""
Tests for Issue #27 — Experiment E3: 50x Flash-Sale Burst.
Validates that:
1. All 6 agents (PPO, DQN, RR, WRR, LC, Threshold) are evaluated on E3 50x burst.
2. 5 repetitions with test seeds 137-141 are executed.
3. 5 result files per agent exist and are not empty (30 total files).
4. Primary hypothesis test: PPO P95 latency < RR P95 latency with p < 0.05.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
from scipy import stats

from evaluation.run_e3_50x import (
    AGENT_NAMES,
    AGENT_ALIASES,
    E3_SCENARIO,
    E3_SEEDS,
    run_e3_experiments,
)


def test_e3_50x_runner_quick(tmp_path):
    """Test runner with truncated steps to verify end-to-end pipeline execution."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ppo_path = root_dir / "models/ppo/best_model.zip"
    dqn_path = root_dir / "models/dqn/best_model.zip"

    summaries, hypothesis_result = run_e3_experiments(
        output_dir=tmp_path,
        seeds=(137,),
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        max_steps=50,
    )

    assert len(summaries) == 6
    for s in summaries:
        assert s["scenario"] == E3_SCENARIO
        assert s["seed"] == 137
        assert s["p95_latency_ms"] > 0
        assert s["throughput_rps"] > 0

        # Check file exists and has rows
        file_path = tmp_path / f"e3_50x_{s['agent']}_seed137.csv"
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_e3_50x_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e3_50x/ and primary hypothesis."""
    results_dir = Path("experiments/results/e3_50x")
    summary_file = results_dir / "e3_50x_summary.csv"
    report_file = results_dir / "hypothesis_test_report.json"

    if not summary_file.exists():
        pytest.skip("Full E3 experiment has not been executed yet.")

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 30, f"Expected 30 summary records (6 agents x 5 seeds), got {len(reader)}"

    agents_found = {row["agent"] for row in reader}
    expected_agents = set(AGENT_NAMES.keys())
    assert agents_found == expected_agents, f"Missing agents: {expected_agents - agents_found}"

    seeds_found = {int(row["seed"]) for row in reader}
    assert seeds_found == set(E3_SEEDS), f"Missing seeds: {set(E3_SEEDS) - seeds_found}"

    # Verify individual run files exist
    for row in reader:
        agent = row["agent"]
        seed = row["seed"]
        agent_file = results_dir / f"e3_50x_{agent}_seed{seed}.csv"
        assert agent_file.exists(), f"Missing result file: {agent_file}"
        assert agent_file.stat().st_size > 0, f"Empty result file: {agent_file}"

    # Verify primary hypothesis test report
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert report["hypothesis_confirmed"] is True
    assert report["p_value"] < 0.05
    assert report["ppo_mean_p95_ms"] < report["rr_mean_p95_ms"]
