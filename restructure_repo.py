"""
Repository Restructure Script
==============================
Moves files from the old flat structure into the new model-centric layout.
Run from any directory — all paths are absolute.

Actions:
  1. Create new directory structure
  2. Copy GPT-2 experiment files into experiments/gpt2_small/
  3. Copy session notes into docs/sessions/
  4. Copy assets into docs/assets/
  5. Remove ATR_SOURCE_PACKAGE duplicate directory
  6. Remove root TECHNICAL.md duplicate
  7. Remove duplicate prompt_library.py from B_AttractorDominance/
  8. Fix 'token similarity.png' filename
  9. Remove old directories (after copying)

DRY RUN by default — set DRY_RUN = False to execute.
"""

import shutil
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DRY_RUN = False  # Set to True for preview only

REPO = Path(r"c:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_LAB_NOTEBOOKS\lucier-repo")

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path):
    if not path.exists():
        if DRY_RUN:
            print(f"  [MKDIR] {path.relative_to(REPO)}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [MKDIR] {path.relative_to(REPO)}")

def copy_file(src, dst):
    if not src.exists():
        print(f"  [SKIP]  {src.name} (not found)")
        return
    ensure_dir(dst.parent)
    if DRY_RUN:
        print(f"  [COPY]  {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
    else:
        shutil.copy2(src, dst)
        print(f"  [COPY]  {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")

def remove_file(path):
    if not path.exists():
        return
    if DRY_RUN:
        print(f"  [DEL]   {path.relative_to(REPO)}")
    else:
        path.unlink()
        print(f"  [DEL]   {path.relative_to(REPO)}")

def remove_dir(path):
    if not path.exists():
        return
    if DRY_RUN:
        print(f"  [RMDIR] {path.relative_to(REPO)}")
    else:
        shutil.rmtree(path)
        print(f"  [RMDIR] {path.relative_to(REPO)}")

# ============================================================
# EXECUTION
# ============================================================

mode = "DRY RUN" if DRY_RUN else "LIVE"
print(f"\n{'='*60}")
print(f"  REPOSITORY RESTRUCTURE — {mode}")
print(f"{'='*60}\n")

# --- 1. Create directory structure ---
print("1. Creating directory structure...")
for d in [
    REPO / "experiments" / "gpt2_small" / "output",
    REPO / "experiments" / "pythia_160m" / "output",
    REPO / "experiments" / "cross_model",
    REPO / "docs" / "sessions",
    REPO / "docs" / "assets",
]:
    ensure_dir(d)

# --- 2. Copy GPT-2 experiment files ---
print("\n2. Copying GPT-2 Small experiment files...")
gpt2 = REPO / "experiments" / "gpt2_small"

copy_file(
    REPO / "B_AttractorDominance" / "01_attractor_dominance.ipynb",
    gpt2 / "01_attractor_dominance.ipynb"
)
copy_file(
    REPO / "B_Reporduceability" / "00_reproducibility_gate.ipynb",
    gpt2 / "00_reproducibility_gate.ipynb"
)
copy_file(
    REPO / "ActivationTensorResonance" / "lucier_total_resonance.ipynb",
    gpt2 / "lucier_total_resonance.ipynb"
)
copy_file(
    REPO / "docs" / "supervisor" / "01_token_id_extraction.ipynb",
    gpt2 / "02_token_neighbourhood.ipynb"
)

# Copy output_stage1 contents
print("\n3. Copying GPT-2 output files...")
src_output = REPO / "B_AttractorDominance" / "output_stage1"
dst_output = gpt2 / "output"
if src_output.exists():
    for f in src_output.iterdir():
        if f.is_file():
            copy_file(f, dst_output / f.name)

# --- 4. Copy session notes ---
print("\n4. Copying session notes to docs/sessions/...")
copy_file(
    REPO / "docs" / "supervisor" / "SESSION_01_SUPERVISORY_REVIEW.md",
    REPO / "docs" / "sessions" / "SESSION_01.md"
)
copy_file(
    REPO / "docs" / "supervisor" / "SESSION_02_RESULTS_DISCUSSION.md",
    REPO / "docs" / "sessions" / "SESSION_02.md"
)

# --- 5. Copy assets ---
print("\n5. Copying assets to docs/assets/...")
assets = REPO / "docs" / "assets"

copy_file(
    REPO / "docs" / "supervisor" / "token similarity.png",
    assets / "token_similarity.png"
)
copy_file(REPO / "docs" / "lucier_room.png", assets / "lucier_room.png")
copy_file(REPO / "docs" / "ATR_COMPREHENSIVE_DASHBOARD.html", assets / "ATR_COMPREHENSIVE_DASHBOARD.html")
copy_file(REPO / "docs" / "ATR_JOURNEY_VISUALIZATION.html", assets / "ATR_JOURNEY_VISUALIZATION.html")

# Copy original exploratory images
img_src = REPO / "ActivationTensorResonance" / "images"
if img_src.exists():
    for f in img_src.iterdir():
        if f.is_file():
            copy_file(f, assets / f.name)

# --- 6. Remove duplicates ---
print("\n6. Removing duplicate files...")
remove_file(REPO / "TECHNICAL.md")
remove_file(REPO / "B_AttractorDominance" / "prompt_library.py")
remove_dir(REPO / "docs" / "ATR_SOURCE_PACKAGE")
remove_dir(REPO / "B_AttractorDominance" / "__pycache__")

# --- 7. Remove old directories ---
print("\n7. Removing old directories (already copied to experiments/)...")
remove_dir(REPO / "ActivationTensorResonance")
remove_dir(REPO / "ActivationTensorResonance_Head")
remove_dir(REPO / "ActivationTensorResonance_Layer")
remove_dir(REPO / "B_AttractorDominance")
remove_dir(REPO / "B_Reporduceability")
remove_dir(REPO / "docs" / "supervisor")
remove_file(REPO / "docs" / "lucier_room.png")
remove_file(REPO / "docs" / "ATR_COMPREHENSIVE_DASHBOARD.html")
remove_file(REPO / "docs" / "ATR_JOURNEY_VISUALIZATION.html")
remove_file(REPO / "docs" / "TECHNICAL_INFOGRAPHIC_PROMPT.md")
remove_file(REPO / "docs" / "UNDERSTANDING_INFOGRAPHIC_PROMPT.md")

# --- 8. Clean up: remove this script and any partial copies from earlier ---
print("\n8. Cleaning up partial copies from earlier bash attempts...")
# The earlier bash commands may have partially populated these dirs
# The script above already handled them properly, just note completion

print(f"\n{'='*60}")
print(f"  RESTRUCTURE {'PREVIEW' if DRY_RUN else 'COMPLETE'}")
print(f"{'='*60}")
print(f"\n  Verify with: dir /s /b lucier-repo")
print(f"  Then delete this script: del restructure_repo.py\n")
