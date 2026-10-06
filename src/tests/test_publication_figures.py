"""
Unit and Integration Tests for Publication Figures.
Phase 7: Results & Statistical Validation - Issue #36
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = ROOT / "experiments" / "results" / "figures"

EXPECTED_FIGURES = [
    "fig1_p95_latency",
    "fig2_throughput_vs_burst",
    "fig3_sla_violation_rate",
    "fig4_reward_convergence",
    "fig5_reward_ablation_heatmap",
]


@pytest.mark.parametrize("fig_name", EXPECTED_FIGURES)
def test_publication_figure_files_exist(fig_name: str):
    """Verify that PNG and PDF versions of each figure exist and are non-empty."""
    png_path = FIGURES_DIR / f"{fig_name}.png"
    pdf_path = FIGURES_DIR / f"{fig_name}.pdf"

    assert png_path.exists(), f"Missing PNG figure: {png_path}"
    assert png_path.stat().st_size > 0, f"PNG figure is empty: {png_path}"

    assert pdf_path.exists(), f"Missing PDF figure: {pdf_path}"
    assert pdf_path.stat().st_size > 0, f"PDF figure is empty: {pdf_path}"


@pytest.mark.parametrize("fig_name", EXPECTED_FIGURES)
def test_png_dpi_and_validity(fig_name: str):
    """Verify that PNG files are valid images and saved at approximately 300 DPI."""
    png_path = FIGURES_DIR / f"{fig_name}.png"
    with Image.open(png_path) as img:
        assert img.width > 500, f"PNG width suspiciously small: {img.width}"
        assert img.height > 400, f"PNG height suspiciously small: {img.height}"
        dpi = img.info.get("dpi")
        if dpi is not None:
            dpi_x, dpi_y = dpi
            assert dpi_x >= 290, f"DPI x should be ~300, got {dpi_x}"
            assert dpi_y >= 290, f"DPI y should be ~300, got {dpi_y}"


@pytest.mark.parametrize("fig_name", EXPECTED_FIGURES)
def test_pdf_format_validity(fig_name: str):
    """Verify that PDF files have valid PDF header for submission."""
    pdf_path = FIGURES_DIR / f"{fig_name}.pdf"
    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header.startswith(b"%PDF-"), f"Invalid PDF header in {pdf_path}: {header}"


def test_generate_publication_figures_pipeline():
    """Verify that scripts/generate_publication_figures.py executes cleanly."""
    import scripts.generate_publication_figures as gpf

    # Run main pipeline
    gpf.main()

    # Re-verify all files exist
    for fig_name in EXPECTED_FIGURES:
        assert (FIGURES_DIR / f"{fig_name}.png").exists()
        assert (FIGURES_DIR / f"{fig_name}.pdf").exists()
