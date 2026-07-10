"""
Update markdown cells in the Pythia notebook with appropriate experiment text.
Code cells are NOT touched.
Run: python fix_pythia_text.py
"""
import json
from pathlib import Path

repo = Path(r"c:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_LAB_NOTEBOOKS\lucier-repo")
nb_path = repo / "experiments" / "pythia_160m" / "01_attractor_dominance.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Replace the first markdown cell (experiment header) with Pythia-appropriate text
new_header = [
    "# Cross-Model ATR: Pythia-160m — Attractor Dominance & Basin Mapping\n",
    "\n",
    "## Experiment Design\n",
    "**This notebook is an EXACT replication of the GPT-2 Small ATR experiment (EXP_009d1),**\n",
    "**applied to EleutherAI's Pythia-160m to test whether attractor basins are model-specific.**\n",
    "\n",
    "### Key Difference: Training Data\n",
    "- **GPT-2 Small:** Trained on *WebText* (Reddit outbound links, 2018)\n",
    "- **Pythia-160m:** Trained on *The Pile* (diverse: books, Wikipedia, GitHub, ArXiv, StackExchange, etc.)\n",
    "\n",
    "### Architecture Comparison\n",
    "| Property | GPT-2 Small | Pythia-160m |\n",
    "| --- | --- | --- |\n",
    "| Parameters | 124M | 85M |\n",
    "| Layers | 12 | 12 |\n",
    "| Heads | 12 | 12 |\n",
    "| d_model | 768 | 768 |\n",
    "| Training Data | WebText (Reddit) | The Pile (diverse) |\n",
    "| Positional Encoding | Learned | Rotary (RoPE) |\n",
    "\n",
    "### Hypotheses Under Test\n",
    "\n",
    "**H_CM1: Basin Existence** — Pythia-160m will also exhibit discrete attractor basins under ATR.\n",
    "(Tests whether attractors are a general transformer property, not GPT-2-specific.)\n",
    "\n",
    "**H_CM2: Basin Divergence** — The terminal basin tokens will DIFFER from GPT-2 Small's basins\n",
    "(`prolet`, `Divine`, `Anarch`, `till`, `solidarity`), reflecting The Pile's different thematic distribution.\n",
    "\n",
    "**H_CM3: Basin Count** — The number of basins may differ, reflecting training corpus diversity.\n",
    "\n",
    "### Method\n",
    "Identical to EXP_009d1. Same 125 prompts, same iteration schedule, same ATR engine.\n",
    "Only the model loaded in STEP 1 is changed.\n",
    "\n",
    "---\n"
]

# Find and replace the first markdown cell
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        cell["source"] = new_header
        print("Updated experiment header markdown.")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated: {nb_path}")
print("Ready to run in Jupyter.")
