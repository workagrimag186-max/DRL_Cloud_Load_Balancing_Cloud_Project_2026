"""
Unit and Integration Tests for Demo Dashboard and 2-Minute JMeter Run.
Phase 9: Demo Dashboard - Issue #44
"""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from metrics.demo_runner import (
    ALGORITHMS,
    BACKEND_CAPACITIES,
    DemoJMeterRunner,
    JTL_PATH,
)
from metrics.visualiser import (
    create_app,
    _render_side_by_side_tab,
    _render_all_algos_tab,
    _render_telemetry_tab,
    _render_jmeter_tab,
)


def test_demo_runner_2_minute_run_and_jtl_export():
    """
    Task: Test on a 2-minute JMeter run (E2 scenario, reduced load) before demo day.
    Verifies that a 120s run completes with all 6 algorithms and exports valid JTL.
    """
    runner = DemoJMeterRunner(duration_sec=120)
    runner.run_full_sync()

    telem = runner.get_latest_telemetry()
    assert telem["is_complete"] is True
    assert telem["current_sec"] == 120
    assert len(telem["algorithms"]) == 6

    # Verify all 6 algorithms are tracked
    for algo in ALGORITHMS:
        assert algo in telem["algorithms"], f"Missing algorithm: {algo}"
        algo_data = telem["algorithms"][algo]
        assert algo_data["p95_latency_ms"] > 0
        assert algo_data["throughput_rps"] > 0
        assert 0.0 <= algo_data["sla_violation_rate"] <= 1.0
        assert len(algo_data["backend_cpus"]) == 4

    # Acceptance Criteria & Primary Claim Verification:
    # Under E2 10x burst, PPO must outperform RoundRobin on P95 latency and SLA violations
    ppo_p95 = telem["algorithms"]["PPO"]["p95_latency_ms"]
    rr_p95 = telem["algorithms"]["RoundRobin"]["p95_latency_ms"]
    assert ppo_p95 < rr_p95, f"PPO P95 ({ppo_p95}ms) should be lower than RR P95 ({rr_p95}ms)"

    # Verify JTL file exists and has valid header and samples
    assert JTL_PATH.exists(), f"JTL file missing: {JTL_PATH}"
    with open(JTL_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 120, f"Expected at least 120 JTL rows, got {len(reader)}"
    assert "timeStamp" in reader[0]
    assert "elapsed" in reader[0]
    assert "responseCode" in reader[0]


def test_dash_app_layout_and_30s_interval():
    """
    Acceptance Criteria:
    Dashboard shows live metrics updating every 30s during demo JMeter run.
    """
    app = create_app()
    assert app is not None
    assert app.title == "FlashBalanceAI — Demo Dashboard"

    # Search for interval component
    interval_comp = None
    for child in app.layout.children:
        if getattr(child, "id", None) == "interval-refresh":
            interval_comp = child
            break

    assert interval_comp is not None, "interval-refresh component must exist"
    assert interval_comp.interval == 30000, f"Default interval must be 30,000 ms (30s), got {interval_comp.interval}"


def test_side_by_side_routing_animation_renders():
    """
    Task: Add side-by-side PPO vs RR routing animation during live JMeter run.
    """
    runner = DemoJMeterRunner(duration_sec=30)
    runner.run_full_sync()

    telem = runner.get_latest_telemetry()
    history = runner.get_history_dataframe()

    tab_elem = _render_side_by_side_tab(telem, history)
    assert tab_elem is not None
    # Verify child structure contains side-by-side routing
    assert len(tab_elem.children) == 3  # Banner, routing columns, animation graph


def test_all_algorithms_comparison_renders():
    """
    Task: compare all algorithms used in the project, add animation for all.
    """
    runner = DemoJMeterRunner(duration_sec=30)
    runner.run_full_sync()

    telem = runner.get_latest_telemetry()
    history = runner.get_history_dataframe()

    tab_elem = _render_all_algos_tab(telem, history)
    assert tab_elem is not None
    assert len(tab_elem.children) == 2  # Distribution graph + Leaderboard table


def test_telemetry_and_jmeter_tabs_render():
    """Verify live updating telemetry graphs and JTL stream render cleanly."""
    runner = DemoJMeterRunner(duration_sec=30)
    runner.run_full_sync()

    telem = runner.get_latest_telemetry()
    history = runner.get_history_dataframe()

    telem_elem = _render_telemetry_tab(telem, history)
    assert telem_elem is not None

    jmeter_elem = _render_jmeter_tab(telem)
    assert jmeter_elem is not None
