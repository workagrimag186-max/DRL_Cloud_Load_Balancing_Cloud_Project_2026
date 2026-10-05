"""
Unit and Integration Tests for Reproducibility Package.
Phase 7: Results & Statistical Validation - Issue #37
"""

import csv
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def test_seed_manifest_structure_and_ranges():
    manifest_path = ROOT / "experiments" / "analysis" / "seed_manifest.csv"
    assert manifest_path.exists(), "seed_manifest.csv must exist"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    assert len(rows) == 100, f"Expected 100 seeds, found {len(rows)}"
    
    required_cols = {"seed", "split", "range_group", "purpose", "target_scenario", "repetition_index", "description"}
    assert required_cols.issubset(set(rows[0].keys())), f"Missing required columns in seed_manifest.csv: {rows[0].keys()}"
    
    train_seeds = [int(r["seed"]) for r in rows if r["split"] == "train"]
    val_seeds = [int(r["seed"]) for r in rows if r["split"] == "val"]
    test_seeds = [int(r["seed"]) for r in rows if r["split"] == "test"]
    
    assert train_seeds == list(range(42, 112)), "Train seeds must span 42-111 inclusive (70 seeds)"
    assert val_seeds == list(range(112, 127)), "Val seeds must span 112-126 inclusive (15 seeds)"
    assert test_seeds == list(range(127, 142)), "Test seeds must span 127-141 inclusive (15 seeds)"
    
    # Check no overlap
    all_seeds = train_seeds + val_seeds + test_seeds
    assert len(all_seeds) == len(set(all_seeds)), "All seeds across train, val, and test must be unique"


def test_requirements_and_environment_pinned():
    req_file = ROOT / "requirements.txt"
    env_file = ROOT / "environment.yml"
    
    assert req_file.exists(), "requirements.txt must exist"
    assert env_file.exists(), "environment.yml must exist"
    
    # Check requirements.txt
    req_text = req_file.read_text(encoding="utf-8").strip().splitlines()
    for line in req_text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "==" in line, f"Package in requirements.txt must be pinned with '==': {line}"
        
    # Check environment.yml
    with open(env_file, "r", encoding="utf-8") as f:
        env_data = yaml.safe_load(f)
        
    assert "dependencies" in env_data, "environment.yml must specify dependencies"
    deps = env_data["dependencies"]
    
    # Verify Python is pinned
    py_pinned = any(isinstance(d, str) and d.startswith("python=") for d in deps)
    assert py_pinned, "Python version must be pinned in environment.yml"


def test_reproduce_scripts_exist():
    sh_script = ROOT / "experiments" / "analysis" / "reproduce_all.sh"
    verify_script = ROOT / "experiments" / "analysis" / "verify_reproduction.py"
    
    assert sh_script.exists(), "reproduce_all.sh must exist"
    assert verify_script.exists(), "verify_reproduction.py must exist"


def test_verify_reproduction_passes():
    from experiments.analysis.verify_reproduction import verify_figures
    assert verify_figures() is True, "verify_figures() must report True within ±2% tolerance"
