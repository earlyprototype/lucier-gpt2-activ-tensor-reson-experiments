"""
Fix the import path in the Pythia notebook so it can find prompt_library.py at repo root.
Run: python fix_pythia_imports.py
"""
import json
from pathlib import Path

repo = Path(r"c:\Users\Fab2\Desktop\AI\_learn\_fold\03_MECHINIPHYLUM\_LAB_NOTEBOOKS\lucier-repo")
nb_path = repo / "experiments" / "pythia_160m" / "01_attractor_dominance.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the cell that imports prompt_library and add sys.path fix before the import
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source_text = "".join(cell["source"])
        if "from prompt_library import" in source_text:
            # Prepend sys.path insert to this cell's source
            fix_lines = [
                "import sys, os\n",
                "sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), '..', '..'))\n",
                "sys.path.insert(0, os.path.abspath(os.path.join('.', '..', '..')))\n",
                "\n",
            ]
            cell["source"] = fix_lines + cell["source"]
            print("Fixed prompt_library import path.")
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Updated: {nb_path}")
print("Ready to run in Jupyter.")
