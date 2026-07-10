"""
Create GPT-2 Medium and Pythia-410m experiment notebooks.
Exact copy of GPT-2 Small code, only model name and markdown text changed.
Run: python create_remaining_notebooks.py
"""
import json, os
from pathlib import Path

repo = Path(r"c:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_LAB_NOTEBOOKS\lucier-repo")
src = repo / "experiments" / "gpt2_small" / "01_attractor_dominance.ipynb"

# Read original notebook once
with open(src, "r", encoding="utf-8") as f:
    nb_template = json.load(f)

MODELS = {
    "gpt2_medium": {
        "dir": "gpt2_medium",
        "tl_name": "gpt2-medium",         # TransformerLens model name
        "old_model": "gpt2-small",          # What to replace
        "display_name": "GPT-2 Medium",
        "params": "345M",
        "layers": 24,
        "heads": 16,
        "d_model": 1024,
        "training_data": "WebText (Reddit outbound links, 2018)",
        "pos_encoding": "Learned",
        "header": [
            "# Cross-Model ATR: GPT-2 Medium — Attractor Dominance & Basin Mapping\n",
            "\n",
            "## Experiment Design\n",
            "**This notebook is an EXACT replication of the GPT-2 Small ATR experiment (EXP_009d1),**\n",
            "**applied to GPT-2 Medium to test whether attractor basin count scales with model capacity.**\n",
            "\n",
            "### Key Difference: Model Scale (SAME training data)\n",
            "- **GPT-2 Small:** 124M params, 12 layers, d_model=768\n",
            "- **GPT-2 Medium:** 345M params, 24 layers, d_model=1024\n",
            "- Both trained on **WebText** (Reddit outbound links, 2018)\n",
            "\n",
            "### Hypotheses Under Test\n",
            "\n",
            "**H_SC1: Basin Existence** — GPT-2 Medium will also exhibit discrete attractor basins under ATR.\n",
            "\n",
            "**H_SC2: Basin Expansion** — With 2.8x more parameters and a larger d_model,\n",
            "GPT-2 Medium may exhibit MORE basins than GPT-2 Small's five,\n",
            "reflecting greater capacity to encode the same training corpus.\n",
            "\n",
            "**H_SC3: Basin Overlap** — Since the training data is identical (WebText),\n",
            "the basin *themes* should overlap with GPT-2 Small (political, theological, etc.),\n",
            "even if specific tokens differ.\n",
            "\n",
            "### Method\n",
            "Identical to EXP_009d1. Same 125 prompts, same iteration schedule, same ATR engine.\n",
            "Only the model loaded in STEP 1 is changed.\n",
            "\n",
            "---\n"
        ]
    },
    "pythia_410m": {
        "dir": "pythia_410m",
        "tl_name": "pythia-410m",
        "old_model": "gpt2-small",
        "display_name": "Pythia-410m",
        "params": "302M",
        "layers": 24,
        "heads": 16,
        "d_model": 1024,
        "training_data": "The Pile (diverse: books, Wikipedia, GitHub, ArXiv, StackExchange, etc.)",
        "pos_encoding": "Rotary (RoPE)",
        "header": [
            "# Cross-Model ATR: Pythia-410m — Attractor Dominance & Basin Mapping\n",
            "\n",
            "## Experiment Design\n",
            "**This notebook is an EXACT replication of the GPT-2 Small ATR experiment (EXP_009d1),**\n",
            "**applied to EleutherAI's Pythia-410m. Fourth corner of the 2×2 factorial design.**\n",
            "\n",
            "### 2×2 Design Context\n",
            "| | WebText (Reddit) | The Pile (diverse) |\n",
            "| --- | --- | --- |\n",
            "| **Small** (~100M, d=768) | GPT-2 Small ✅ | Pythia-160m |\n",
            "| **Medium** (~300M, d=1024) | GPT-2 Medium | **Pythia-410m** ← this |\n",
            "\n",
            "### Key Difference: Scale + Different Training Data\n",
            "- **Pythia-160m:** 85M params, 12 layers, d_model=768, The Pile\n",
            "- **Pythia-410m:** 302M params, 24 layers, d_model=1024, The Pile\n",
            "\n",
            "### Hypotheses Under Test\n",
            "\n",
            "**H_4C1: Scale Replication** — Does basin count scale with capacity on The Pile,\n",
            "replicating the GPT-2 Small → Medium scaling result?\n",
            "\n",
            "**H_4C2: Corpus Replication** — Do Pythia-410m basins match GPT-2 Medium basins\n",
            "(same scale, different data), or Pythia-160m basins (same data, different scale)?\n",
            "\n",
            "**H_4C3: Interaction Effect** — Is the attractor landscape driven primarily by\n",
            "training data, model scale, or their interaction?\n",
            "\n",
            "### Method\n",
            "Identical to EXP_009d1. Same 125 prompts, same iteration schedule, same ATR engine.\n",
            "Only the model loaded in STEP 1 is changed.\n",
            "\n",
            "---\n"
        ]
    }
}

for key, cfg in MODELS.items():
    print(f"\n--- Creating {cfg['display_name']} notebook ---")

    # Deep copy
    nb = json.loads(json.dumps(nb_template))

    # 1. Clear outputs
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    # 2. Replace model name in ALL cells (code + markdown)
    for cell in nb["cells"]:
        cell["source"] = [
            line.replace(cfg["old_model"], cfg["tl_name"])
                .replace("GPT-2 Small", cfg["display_name"])
                .replace('"output_stage1"', '"output"')
            for line in cell["source"]
        ]

    # 3. Replace first markdown cell with experiment-specific header
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = cfg["header"]
            break

    # 4. Fix prompt_library import path
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source_text = "".join(cell["source"])
            if "from prompt_library import" in source_text:
                fix_lines = [
                    "import sys, os\n",
                    "sys.path.insert(0, os.path.abspath(os.path.join('.', '..', '..')))\n",
                    "\n",
                ]
                cell["source"] = fix_lines + cell["source"]
                break

    # 5. Create output dir and save
    out_dir = repo / "experiments" / cfg["dir"]
    (out_dir / "output").mkdir(parents=True, exist_ok=True)

    dst = out_dir / "01_attractor_dominance.ipynb"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"  Created: {dst}")
    print(f"  Size: {dst.stat().st_size:,} bytes")
    print(f"  Model: {cfg['tl_name']}")

print("\n✅ All notebooks created. Open each in Jupyter and Run All Cells.")
print("   GPT-2 Medium: ~1.4GB download, ~90 mins on CPU")
print("   Pythia-410m:  ~1.2GB download, ~90 mins on CPU")
