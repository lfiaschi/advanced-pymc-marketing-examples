"""
Extract and run notebook cells to understand content and generate figures
"""
import json
from pathlib import Path
import re

# Load notebook
with open('notebooks/05_comprehensive_prior_sensitivity.ipynb', 'r') as f:
    notebook = json.load(f)

# Extract code cells
code_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.strip():
            code_cells.append(source)

# Write out complete script
output_path = Path('notebooks/05_comprehensive_prior_sensitivity_script.py')
with open(output_path, 'w') as f:
    # Add header
    f.write("#!/usr/bin/env python\n")
    f.write("# Extracted from 05_comprehensive_prior_sensitivity.ipynb\n\n")

    # Write all code cells
    for i, code in enumerate(code_cells):
        f.write(f"# Cell {i+1}\n")
        f.write(code)
        f.write("\n\n")

print(f"Script written to {output_path}")
print(f"Total code cells extracted: {len(code_cells)}")

# Also extract markdown cells for context
markdown_cells = []
for cell in notebook['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if source.strip():
            markdown_cells.append(source)

# Look for conclusions
print("\n=== Looking for conclusions ===")
for i, md in enumerate(markdown_cells):
    if 'conclusion' in md.lower() or 'summary' in md.lower() or 'takeaway' in md.lower():
        print(f"\nMarkdown cell {i}:")
        print(md[:500] + "..." if len(md) > 500 else md)