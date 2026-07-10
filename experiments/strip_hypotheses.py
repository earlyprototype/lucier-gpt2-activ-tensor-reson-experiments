"""
Strip hypothesis predictions from all hypothesis_assessment.md files.

Removes:
  - The 'Predicted' column
  - The 'Match?' column  
  - The entire 'Prediction Mismatches' section
  - Renames title from 'Hypothesis Assessment' to 'Basin Assessment'

Run from: lucier-repo/experiments/
"""
import os

FILES = [
    'gpt2_small/output/hypothesis_assessment.md',
    'gpt2_medium/output/hypothesis_assessment.md',
    'pythia_410m/output/hypothesis_assessment.md',
    'gpt2_small/pythia_160m/output/hypothesis_assessment.md',
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

for fpath in FILES:
    full_path = os.path.join(SCRIPT_DIR, fpath)
    if not os.path.exists(full_path):
        print(f'[SKIP] {fpath} not found')
        continue

    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Rename the title
    content = content.replace(
        '# Stage 1 Results: Hypothesis Assessment',
        '# Stage 1 Results: Basin Assessment'
    )

    # Fix table headers
    content = content.replace(
        '| Prompt | Category | Predicted | Actual Terminal | Match? |',
        '| Prompt | Category | Terminal Basin |'
    )
    content = content.replace(
        '|:---|:---|:---|:---|:---|',
        '|:---|:---|:---|'
    )

    lines = content.split('\n')
    new_lines = []
    skip_mismatches = False

    for line in lines:
        # Skip the entire Prediction Mismatches section
        if line.strip().startswith('## Prediction Mismatches'):
            skip_mismatches = True
            continue
        if skip_mismatches:
            if line.strip().startswith('## ') and 'Prediction Mismatches' not in line:
                skip_mismatches = False
                new_lines.append(line)
            continue

        # Strip Predicted and Match columns from data rows
        if line.startswith('|') and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            # 5-col table: ['', 'Prompt', 'Category', 'Predicted', 'Actual', 'Match', '']
            if len(parts) == 7:
                # Keep only Prompt (1), Category (2), Actual Terminal (4)
                new_line = f'| {parts[1]} | {parts[2]} | {parts[4]} |'
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    result = '\n'.join(new_lines).rstrip() + '\n'

    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f'[DONE] {fpath}')

print('\nAll files processed.')
