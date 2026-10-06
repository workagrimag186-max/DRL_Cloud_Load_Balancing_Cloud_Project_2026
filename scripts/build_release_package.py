"""
FlashBalanceAI — v1.0 Release Packaging Script.
Phase 9: Demo, Reproducibility & Teardown - Issue #45

Assembles all deliverables into dist/ for GitHub Release attachment:
  - Publication figures (PNG + PDF 300 dpi)
  - Statistical analysis summaries (p-values & 95% CI)
  - Trained DRL agent checkpoints (PPO & DQN)
  - Reproducibility automation scripts
  - Release metadata and checksums
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"
RELEASE_ZIP = DIST_DIR / "FlashBalanceAI-v1.0-Reproducibility-Package.zip"


def calculate_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_release_bundle():
    print("=======================================================")
    print("FlashBalanceAI — Assembling v1.0 Release Deliverables")
    print("=======================================================")

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect files to package
    files_to_pack = [
        # Figures
        *list((ROOT / "experiments" / "results" / "figures").glob("*.pdf")),
        *list((ROOT / "experiments" / "results" / "figures").glob("*.png")),
        # Analysis
        ROOT / "experiments" / "analysis" / "statistical_tests.csv",
        ROOT / "experiments" / "analysis" / "summary_statistics.csv",
        ROOT / "experiments" / "analysis" / "seed_manifest.csv",
        ROOT / "experiments" / "analysis" / "reproduce_all.ps1",
        ROOT / "experiments" / "analysis" / "reproduce_all.sh",
        ROOT / "experiments" / "analysis" / "verify_reproduction.py",
        # Models
        ROOT / "models" / "ppo" / "best_model.zip",
        ROOT / "models" / "dqn" / "best_model.zip",
        # Documentation & Config
        ROOT / "GITHUB_PROJECT_PLAN.md",
        ROOT / "README.md",
        ROOT / "RELEASE_NOTES_v1.0.md",
        ROOT / "requirements.txt",
        ROOT / "environment.yml",
    ]

    existing_files = [f for f in files_to_pack if f.exists()]
    print(f"Discovered {len(existing_files)} deliverable files to package.")

    # 2. Create Zip Package
    manifest = []
    with zipfile.ZipFile(RELEASE_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in existing_files:
            rel_path = file.relative_to(ROOT)
            zipf.write(file, arcname=str(rel_path))
            sha = calculate_sha256(file)
            manifest.append(f"{sha}  {rel_path.as_posix()}")
            print(f"  + Added: {rel_path.as_posix()} ({file.stat().st_size:,} bytes)")

    # 3. Write SHA256 checksums manifest
    checksums_path = DIST_DIR / "CHECKSUMS-v1.0.txt"
    with open(checksums_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")

    print(f"\nCreated Release Bundle: {RELEASE_ZIP} ({RELEASE_ZIP.stat().st_size:,} bytes)")
    print(f"Created Checksums: {checksums_path}")
    print("=======================================================")
    print("Release package assembly completed successfully.")


if __name__ == "__main__":
    build_release_bundle()
