"""
Tests for Issue #25 — Experiment E1: Baseline Traffic (1x Burst).
Validates that:
1. All 6 agents (PPO, DQN, RR, WRR, LC, Threshold) are evaluated on E1 Baseline.
2. 5 repetitions with test seeds 127-131 are executed.
3. 5 result files per agent exist and are not empty.
4. Acceptance criteria: P95 latency < 200ms for all agents across all seeds.
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from evaluation.run_e1_baseline import (
    AGENT_NAMES,
    AGENT_ALIASES,
    E1_SCENARIO,
    E1_SEEDS,
    run_e1_experiments,
)


def test_e1_baseline_runner_quick(tmp_path):
    """Test runner with truncated steps to verify end-to-end pipeline execution."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ppo_path = root_dir / "models/ppo/best_model.zip"
    dqn_path = root_dir / "models/dqn/best_model.zip"

    summaries, _ = run_e1_experiments(
        output_dir=tmp_path,
        seeds=(127,),
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        max_steps=50,
    )

    assert len(summaries) == 6
    for s in summaries:
        assert s["scenario"] == E1_SCENARIO
        assert s["seed"] == 127
        assert s["p95_latency_ms"] < 200.0
        assert s["sla_violation_rate"] == 0.0

        # Check file exists and has rows
        file_path = tmp_path / f"e1_baseline_{s['agent']}_seed127.csv"
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_e1_baseline_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e1_baseline/ meet acceptance criteria."""
    results_dir = Path("experiments/results/e1_baseline")
    summary_file = results_dir / "e1_baseline_summary.csv"

    # If results haven't been run yet, skip or assert existence
    if not summary_file.exists():
        pytest.skip("Full E1 baseline experiment has not been executed yet.")

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 30, f"Expected 30 summary records (6 agents x 5 seeds), got {len(reader)}"

    agents_found = {row["agent"] for row in reader}
    expected_agents = set(AGENT_NAMES.keys())
    assert agents_found == expected_agents, f"Missing agents: {expected_agents - agents_found}"

    seeds_found = {int(row["seed"]) for row in reader}
    assert seeds_found == set(E1_SEEDS), f"Missing seeds: {set(E1_SEEDS) - seeds_found}"

    for row in reader:
        p95 = float(row["p95_latency_ms"])
        agent = row["agent"]
        seed = row["seed"]
        assert p95 < 200.0, f"Acceptance criteria failed: {agent} seed {seed} P95 {p95}ms >= 200ms"

        # Check that individual result file exists for each agent and seed
        agent_file = results_dir / f"e1_baseline_{agent}_seed{seed}.csv"
        assert agent_file.exists(), f"Missing result file: {agent_file}"
        assert agent_file.stat().st_size > 0, f"Empty result file: {agent_file}"
