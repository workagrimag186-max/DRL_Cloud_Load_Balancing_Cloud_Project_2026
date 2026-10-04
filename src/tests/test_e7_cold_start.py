"""
Tests for Issue #31 — Experiment E7: Cold-Start Latency.
Validates that:
1. Inference server startup latency (Lambda invocation to first action) is measured across 5 reps.
2. Action selection latency per call is measured in steady state.
3. 5 result files exist per tested agent (10 total trace files) and are non-empty.
4. Latencies comply with the SLA budget (< 500ms).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from evaluation.run_e7_cold_start import (
    E7_REPS,
    SLA_BUDGET_MS,
    run_e7_experiments,
)


def test_e7_cold_start_runner_quick(tmp_path):
    """Test runner with 1 repetition and 5 warm calls to verify quick execution."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ppo_path = root_dir / "models/ppo/best_model.zip"
    dqn_path = root_dir / "models/dqn/best_model.zip"

    summaries, report = run_e7_experiments(
        output_dir=tmp_path,
        reps=(1,),
        seeds=(152,),
        ppo_path=ppo_path,
        dqn_path=dqn_path,
        num_warm_calls=5,
    )

    assert len(summaries) == 2  # PPO and DQN
    for s in summaries:
        assert s["rep"] == 1
        assert s["cold_start_total_ms"] > 0
        assert s["warm_mean_ms"] > 0
        assert s["warm_p95_ms"] > 0
        assert s["sla_budget_ms"] == SLA_BUDGET_MS
        assert s["warm_sla_compliant_rate"] == 1.0

        # Check trace file exists
        trace_file = tmp_path / f"e7_cold_start_{s['agent']}_rep1.csv"
        assert trace_file.exists()
        assert trace_file.stat().st_size > 0


def test_e7_cold_start_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e7_cold_start/ and SLA compliance."""
    results_dir = Path("experiments/results/e7_cold_start")
    summary_file = results_dir / "e7_cold_start_summary.csv"
    report_file = results_dir / "cold_start_report.json"

    if not summary_file.exists():
        pytest.skip("Full E7 experiment has not been executed yet.")

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 10, f"Expected 10 summary records (2 agents x 5 reps), got {len(reader)}"

    agents_found = {row["agent"] for row in reader}
    assert agents_found == {"PPO", "DQN"}, f"Unexpected agents: {agents_found}"

    reps_found = {int(row["rep"]) for row in reader}
    assert reps_found == set(E7_REPS), f"Missing reps: {set(E7_REPS) - reps_found}"

    # Verify individual trace files exist
    for row in reader:
        agent = row["agent"]
        rep = row["rep"]
        trace_file = results_dir / f"e7_cold_start_{agent}_rep{rep}.csv"
        assert trace_file.exists(), f"Missing trace file: {trace_file}"
        assert trace_file.stat().st_size > 0, f"Empty trace file: {trace_file}"

    # Verify SLA report
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    assert report["ppo"]["cold_start_sla_pass"] is True
    assert report["ppo"]["warm_sla_pass"] is True
    assert report["ppo"]["mean_warm_p95_ms"] < SLA_BUDGET_MS
    assert report["dqn"]["cold_start_sla_pass"] is True
    assert report["dqn"]["warm_sla_pass"] is True
    assert report["dqn"]["mean_warm_p95_ms"] < SLA_BUDGET_MS
