"""
Create Pythia-160m notebook by EXACT copy of GPT-2 notebook.
Only changes: model name string + clear outputs.
Run: python create_pythia_notebook.py
"""
import json
from pathlib import Path

repo = Path(r"c:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_LAB_NOTEBOOKS\lucier-repo")
src = repo / "experiments" / "gpt2_small" / "01_attractor_dominance.ipynb"
dst = repo / "experiments" / "pythia_160m" / "01_attractor_dominance.ipynb"

# 1. Read original notebook
with open(src, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 2. Clear all outputs (fresh notebook)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# 3. Replace ONLY the model name in source lines
for cell in nb["cells"]:
    cell["source"] = [
        line.replace("gpt2-small", "pythia-160m").replace("GPT-2 Small", "Pythia-160m")
        for line in cell["source"]
    ]

# 4. Also need to update output dir name so results don't clash
for cell in nb["cells"]:
    cell["source"] = [
        line.replace('"output_stage1"', '"output"')
        for line in cell["source"]
    ]

# 5. Save
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created: {dst}")
print(f"Size: {dst.stat().st_size:,} bytes")
print("Open it in Jupyter and run all cells.")
