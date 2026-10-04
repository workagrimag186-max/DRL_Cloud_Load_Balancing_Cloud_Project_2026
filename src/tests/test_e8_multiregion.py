"""
Tests for Issue #32 — Experiment E8: Multi-Region Simulation.
Validates that:
1. All 6 agents (PPO, DQN, RR, WRR, LC, Threshold) are evaluated under multi-region outage conditions.
2. 5 repetitions with test seeds 157-161 are executed.
3. 5 result files per agent exist and are not empty (30 total files).
4. Verifies PPO adaptation against RoundRobin: PPO P95 latency < RR P95 latency with p < 0.05.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
from scipy import stats

from evaluation.run_e8_multiregion import (
    AGENT_NAMES,
    AGENT_ALIASES,
    E8_SCENARIO,
    E8_SEEDS,
    run_e8_experiments,
)


def test_e8_multiregion_runner_quick(tmp_path):
    """Test runner with truncated steps to verify end-to-end pipeline execution."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ppo_path = root_dir / "models/ppo/best_model.zip"
    dqn_path = root_dir / "models/dqn/best_model.zip"

    summaries, hypothesis_result = run_e8_experiments(
        output_dir=tmp_path,
        seeds=(157,),
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        max_steps=50,
    )

    assert len(summaries) == 6
    for s in summaries:
        assert s["scenario"] == E8_SCENARIO
        assert s["seed"] == 157
        assert s["p95_latency_ms"] > 0
        assert s["throughput_rps"] > 0

        # Check file exists and has rows
        file_path = tmp_path / f"e8_multiregion_{s['agent']}_seed157.csv"
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_e8_multiregion_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e8_multiregion/ and adaptation hypotheses."""
    results_dir = Path("experiments/results/e8_multiregion")
    summary_file = results_dir / "e8_multiregion_summary.csv"
    report_file = results_dir / "hypothesis_test_report.json"

    if not summary_file.exists():
        pytest.skip("Full E8 experiment has not been executed yet.")

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 30, f"Expected 30 summary records (6 agents x 5 seeds), got {len(reader)}"

    agents_found = {row["agent"] for row in reader}
    expected_agents = set(AGENT_NAMES.keys())
    assert agents_found == expected_agents, f"Missing agents: {expected_agents - agents_found}"

    seeds_found = {int(row["seed"]) for row in reader}
    assert seeds_found == set(E8_SEEDS), f"Missing seeds: {set(E8_SEEDS) - seeds_found}"

    # Verify individual run files exist
    for row in reader:
        agent = row["agent"]
        seed = row["seed"]
        agent_file = results_dir / f"e8_multiregion_{agent}_seed{seed}.csv"
        assert agent_file.exists(), f"Missing result file: {agent_file}"
        assert agent_file.stat().st_size > 0, f"Empty result file: {agent_file}"

    # Verify report
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert "ppo_vs_rr" in report
    assert report["ppo_vs_rr"]["confirmed"] is True
    assert report["ppo_vs_rr"]["p_value"] < 0.05
    assert report["ppo_vs_rr"]["ppo_mean_p95_ms"] < report["ppo_vs_rr"]["rr_mean_p95_ms"]
