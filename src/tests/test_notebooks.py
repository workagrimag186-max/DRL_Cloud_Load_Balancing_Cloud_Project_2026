"""
Unit tests for Jupyter notebooks validation.
Phase 7: Results & Statistical Validation - Issue #38
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent

NOTEBOOK_NAMES = [
    "01_traffic_analysis.ipynb",
    "02_ppo_training.ipynb",
    "03_results_analysis.ipynb",
    "04_ablation_study.ipynb",
]

NOTEBOOK_DIRS = [
    ROOT / "feature" / "AgrimaGupta" / "notebooks",
    ROOT / "notebooks",
]


@pytest.mark.parametrize("nb_dir", NOTEBOOK_DIRS)
@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebooks_exist_and_valid_json(nb_dir: Path, name: str):
    nb_path = nb_dir / name
    assert nb_path.exists(), f"Notebook {nb_path} must exist"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    assert "cells" in nb_data, f"Notebook {name} missing cells"
    assert len(nb_data["cells"]) > 0, f"Notebook {name} must have cells"


@pytest.mark.parametrize("nb_dir", NOTEBOOK_DIRS)
@pytest.mark.parametrize("name", NOTEBOOK_NAMES)
def test_notebooks_executed_with_zero_errors(nb_dir: Path, name: str):
    nb_path = nb_dir / name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
        
    code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) > 0, f"Notebook {name} must contain code cells"
    
    for idx, cell in enumerate(code_cells, start=1):
        assert cell.get("execution_count") is not None, (
            f"Code cell {idx} in {name} has not been executed (execution_count is None)"
        )
        outputs = cell.get("outputs", [])
        for out in outputs:
            assert out.get("output_type") != "error", (
                f"Error detected in executed cell {idx} of {name}: {out.get('ename')}: {out.get('evalue')}"
            )
