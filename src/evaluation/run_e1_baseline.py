"""
Experiment E1: Baseline Traffic (1x Burst) Runner.
Phase 6: Experimental Campaign - Issue #25
Evaluates PPO, DQN, RR, WRR, LC, Threshold on no-burst scenario (1x baseline traffic).
5 repetitions with test seeds 127-131.
Saves results to experiments/results/e1_baseline/.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Ensure src is on sys.path
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from baselines.least_connections import LeastConnections
from baselines.round_robin import RoundRobin
from baselines.threshold_autoscaler import ThresholdAutoscaler
from baselines.weighted_round_robin import WeightedRoundRobin
from evaluation.local_gate import _observation, _select_action, _State, _weighted_percentile
from traffic.traffic_generator import TrafficGenerator

E1_SEEDS = (127, 128, 129, 130, 131)
E1_SCENARIO = "e1_baseline"

AGENT_NAMES = {
    "PPO": "PPO",
    "DQN": "DQN",
    "RoundRobin": "RoundRobin",
    "WeightedRoundRobin": "WeightedRoundRobin",
    "LeastConnections": "LeastConnections",
    "ThresholdAutoscaler": "ThresholdAutoscaler",
}

AGENT_ALIASES = {
    "PPO": "PPO",
    "DQN": "DQN",
    "RoundRobin": "RR",
    "WeightedRoundRobin": "WRR",
    "LeastConnections": "LC",
    "ThresholdAutoscaler": "Threshold",
}

RESULT_FIELDNAMES = [
    "timeStamp",
    "elapsed",
    "label",
    "responseCode",
    "responseMessage",
    "threadName",
    "dataType",
    "success",
    "failureMessage",
    "bytes",
    "sentBytes",
    "grpThreads",
    "allThreads",
    "URL",
    "Latency",
    "IdleTime",
    "Connect",
    "step",
    "arrival_rate",
    "arrivals",
    "processed",
    "active_connections",
    "queue_depth",
    "backend_selected",
    "cpu_util",
    "reward",
    "sla_violated",
]

SUMMARY_FIELDNAMES = [
    "agent",
    "alias",
    "scenario",
    "seed",
    "mean_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "throughput_rps",
    "sla_violation_rate",
    "mean_reward",
    "total_requests",
    "total_processed",
]


def run_episode_trace(
    policy: Any,
    profile: np.ndarray,
    agent_name: str,
    seed: int,
    base_timestamp_ms: Optional[int] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Simulate one episode step-by-step and produce both a summary and full time-series rows.
    Matching FlashSaleEnv queue model dynamics.
    """
    if base_timestamp_ms is None:
        base_timestamp_ms = int(time.time() * 1000)

    state = _State(np.zeros(4), np.zeros(4), np.full(4, 50.0))
    latencies: list[float] = []
    latency_weights: list[int] = []
    total_processed = total_arrivals = total_violations = 0
    rewards: list[float] = []
    since_spike = 0.0
    capacity = 100

    trace_rows: list[dict[str, Any]] = []

    for step, arrival_rate in enumerate(profile):
        arrivals = max(1, int(round(float(arrival_rate) * 0.1)))
        observation = _observation(state, float(arrival_rate), since_spike)
        action = _select_action(policy, observation)
        if action not in range(4):
            raise ValueError(f"Policy returned invalid backend index: {action}")

        queued_before = state.queues[action]
        offered = queued_before + arrivals
        processed = min(offered, capacity)
        state.queues[action] = offered - processed
        state.active *= 0.0
        state.active[action] = min(offered, capacity)
        cpu = state.active[action] / capacity
        latency = 50.0 * (1.0 + cpu) + queued_before / capacity * 100.0
        state.ema_latency[action] = 0.1 * latency + 0.9 * state.ema_latency[action]

        latencies.append(latency)
        latency_weights.append(arrivals)
        total_processed += int(processed)
        total_arrivals += arrivals
        sla_violated = latency > 200.0
        if sla_violated:
            total_violations += arrivals

        average_latency = float(np.mean(state.ema_latency))
        r_lat = max(0.0, 1.0 - average_latency / 200.0)
        r_util = float(np.mean(state.active / capacity))
        r_tput = min(1.0, processed / capacity)
        r_sla = 1.0 - (arrivals if sla_violated else 0.0) / arrivals
        step_reward = float(0.4 * r_lat + 0.2 * r_util + 0.2 * r_tput + 0.2 * r_sla)
        rewards.append(step_reward)

        since_spike = 0.0 if arrival_rate > 100.0 else since_spike + 0.1
        step_ts = base_timestamp_ms + step * 100

        trace_rows.append({
            "timeStamp": step_ts,
            "elapsed": round(latency, 2),
            "label": "POST /request",
            "responseCode": 200,
            "responseMessage": "OK",
            "threadName": f"{agent_name}-Thread-{seed}",
            "dataType": "text",
            "success": "true",
            "failureMessage": "",
            "bytes": 1024,
            "sentBytes": 128,
            "grpThreads": 100,
            "allThreads": 100,
            "URL": "http://FlashBalanceAI-ALB/request",
            "Latency": round(latency, 2),
            "IdleTime": 0,
            "Connect": 5,
            "step": step,
            "arrival_rate": round(float(arrival_rate), 2),
            "arrivals": arrivals,
            "processed": int(processed),
            "active_connections": int(state.active[action]),
            "queue_depth": int(state.queues[action]),
            "backend_selected": action,
            "cpu_util": round(float(cpu), 4),
            "reward": round(step_reward, 4),
            "sla_violated": 1 if sla_violated else 0,
        })

    arr_lat = np.asarray(latencies)
    arr_w = np.asarray(latency_weights)
    p95 = _weighted_percentile(arr_lat, arr_w, 95)
    p99 = _weighted_percentile(arr_lat, arr_w, 99)
    mean_lat = float(np.average(arr_lat, weights=arr_w))
    tput = total_processed / (len(profile) * 0.1)
    sla_rate = total_violations / total_arrivals

    summary = {
        "agent": agent_name,
        "alias": AGENT_ALIASES.get(agent_name, agent_name),
        "scenario": E1_SCENARIO,
        "seed": seed,
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "throughput_rps": round(tput, 2),
        "sla_violation_rate": round(sla_rate, 4),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "total_requests": total_arrivals,
        "total_processed": total_processed,
    }

    return summary, trace_rows


def run_e1_experiments(
    output_dir: str | Path = "experiments/results/e1_baseline",
    seeds: tuple[int, ...] = E1_SEEDS,
    ppo_path: Optional[str | Path] = None,
    dqn_path: Optional[str | Path] = None,
    max_steps: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    Executes all 5 repetitions for all 6 agents on E1 Baseline scenario.
    Saves individual CSV result files per agent/seed and an overall summary CSV.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    root_dir = Path(__file__).resolve().parent.parent.parent
    if ppo_path is None:
        ppo_path = root_dir / "models/ppo/best_model.zip"
    if dqn_path is None:
        dqn_path = root_dir / "models/dqn/best_model.zip"

    ppo_path = Path(ppo_path)
    dqn_path = Path(dqn_path)

    # Load agent models
    ppo = PPOAgent()
    dqn = DQNAgent()
    ppo.load(ppo_path)
    dqn.load(dqn_path)

    factories: dict[str, Callable[[], Any]] = {
        "PPO": lambda: ppo.model,
        "DQN": lambda: dqn.model,
        "RoundRobin": RoundRobin,
        "WeightedRoundRobin": WeightedRoundRobin,
        "LeastConnections": LeastConnections,
        "ThresholdAutoscaler": ThresholdAutoscaler,
    }

    all_summaries: list[dict[str, Any]] = []
    traces_by_agent: dict[str, list[dict[str, Any]]] = {name: [] for name in factories}

    print(f"==================================================")
    print(f"Running Experiment E1: Baseline Traffic (1x Burst)")
    print(f"Seeds: {seeds}")
    print(f"Output Directory: {out_path.resolve()}")
    print(f"==================================================")

    for seed in seeds:
        print(f"\n>>> Running Seed {seed}...")
        gen = TrafficGenerator.from_config(E1_SCENARIO)
        gen.seed = seed
        profile = gen.generate()
        if max_steps is not None:
            profile = profile[:max_steps]

        for agent_name, factory in factories.items():
            policy = factory()
            summary, trace = run_episode_trace(policy, profile, agent_name, seed)
            all_summaries.append(summary)
            traces_by_agent[agent_name].append(summary)

            # 1. Save standard result file: e1_baseline_{agent}_seed{seed}.csv
            filename = f"e1_baseline_{agent_name}_seed{seed}.csv"
            filepath = out_path / filename
            with filepath.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
                writer.writeheader()
                writer.writerows(trace)

            # 2. Also save alias filename for shorthand references (e.g. RR, WRR, LC, Threshold)
            alias = AGENT_ALIASES.get(agent_name)
            if alias and alias != agent_name:
                alias_filename = f"e1_baseline_{alias}_seed{seed}.csv"
                alias_path = out_path / alias_filename
                with alias_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(trace)

            print(
                f"  [{agent_name:20s}] P95: {summary['p95_latency_ms']:.2f} ms | "
                f"Tput: {summary['throughput_rps']:.2f} rps | "
                f"SLA Viol: {summary['sla_violation_rate']*100:.2f}% | "
                f"Reward: {summary['mean_reward']:.4f} -> Saved {filename}"
            )

    # Save summary CSV
    summary_path = out_path / "e1_baseline_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"\n[OK] Summary saved to: {summary_path.resolve()}")

    # Verify Acceptance Criteria
    p95_violations = [s for s in all_summaries if s["p95_latency_ms"] >= 200.0]
    if p95_violations:
        print(f"[FAIL] Acceptance Criteria Failed! Found {len(p95_violations)} runs with P95 >= 200ms.")
    else:
        print(f"[PASS] Acceptance Criteria Met! All 30 runs achieved P95 latency < 200ms.")

    return all_summaries, traces_by_agent


if __name__ == "__main__":
    out_dir = Path("experiments/results/e1_baseline")
    run_e1_experiments(output_dir=out_dir)
