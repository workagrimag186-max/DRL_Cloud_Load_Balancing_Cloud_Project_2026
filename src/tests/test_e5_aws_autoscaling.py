"""
Tests for Issue #29 — Experiment E5: AWS Auto Scaling Comparison.
Validates that:
1. All 6 configurations (PPO_with_ASG, PPO_without_ASG, Threshold_ASG, RR_without_ASG, PPO_reactive_ASG, RR_with_ASG) are evaluated on E2 scenario.
2. 5 repetitions with test seeds 132-136 are executed.
3. 5 result files per configuration exist and are not empty (30 total files).
4. Quantifies DRL value beyond standard AWS autoscaling: PPO_with_ASG < Threshold_ASG (p < 0.05).
5. Quantifies ASG value to PPO: PPO_with_ASG < PPO_without_ASG (p < 0.05).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
from scipy import stats

from evaluation.run_e5_aws_autoscaling import (
    CONFIG_NAMES,
    CONFIG_ALIASES,
    E5_SCENARIO,
    E5_SEEDS,
    run_e5_experiments,
)


def test_e5_aws_autoscaling_runner_quick(tmp_path):
    """Test runner with truncated steps to verify end-to-end pipeline execution."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    ppo_path = root_dir / "models/ppo/best_model.zip"

    summaries, hypothesis_result = run_e5_experiments(
        output_dir=tmp_path,
        seeds=(132,),
        ppo_path=ppo_path,
        max_steps=50,
    )

    assert len(summaries) == 6
    for s in summaries:
        assert s["scenario"] == E5_SCENARIO
        assert s["seed"] == 132
        assert s["p95_latency_ms"] > 0
        assert s["throughput_rps"] > 0

        # Check file exists and has rows
        file_path = tmp_path / f"e5_aws_autoscaling_{s['config']}_seed132.csv"
        assert file_path.exists()
        assert file_path.stat().st_size > 0


def test_e5_aws_autoscaling_saved_results_compliance():
    """Verify saved experimental results in experiments/results/e5_aws_autoscaling/ and hypotheses."""
    results_dir = Path("experiments/results/e5_aws_autoscaling")
    summary_file = results_dir / "e5_aws_autoscaling_summary.csv"
    report_file = results_dir / "hypothesis_test_report.json"

    if not summary_file.exists():
        pytest.skip("Full E5 experiment has not been executed yet.")

    with summary_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 30, f"Expected 30 summary records (6 configs x 5 seeds), got {len(reader)}"

    configs_found = {row["config"] for row in reader}
    expected_configs = set(CONFIG_NAMES.keys())
    assert configs_found == expected_configs, f"Missing configs: {expected_configs - configs_found}"

    seeds_found = {int(row["seed"]) for row in reader}
    assert seeds_found == set(E5_SEEDS), f"Missing seeds: {set(E5_SEEDS) - seeds_found}"

    # Verify individual run files exist
    for row in reader:
        config = row["config"]
        seed = row["seed"]
        config_file = results_dir / f"e5_aws_autoscaling_{config}_seed{seed}.csv"
        assert config_file.exists(), f"Missing result file: {config_file}"
        assert config_file.stat().st_size > 0, f"Empty result file: {config_file}"

    # Verify hypothesis report
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))

    # 1. PPO with ASG vs Standard AWS Threshold Auto Scaling
    drl_report = report["drl_beyond_autoscaling"]
    assert drl_report["confirmed"] is True
    assert drl_report["p_value"] < 0.05
    assert drl_report["ppo_with_asg_mean_p95_ms"] < drl_report["threshold_asg_mean_p95_ms"]

    # 2. ASG Value to PPO
    asg_report = report["asg_value_to_ppo"]
    assert asg_report["confirmed"] is True
    assert asg_report["p_value"] < 0.05
    assert asg_report["ppo_with_asg_mean_p95_ms"] < asg_report["ppo_without_asg_mean_p95_ms"]
