"""
Script to export all visualizations from notebook 07_chronos_pymc_marketing.ipynb
"""

import json
import base64
from pathlib import Path
import re

def extract_plots_from_notebook(notebook_path: Path, output_dir: Path) -> None:
    """Extract all plots from a notebook and save them as PNG files."""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    plot_count = 0

    # Iterate through cells
    for cell_idx, cell in enumerate(notebook['cells']):
        if 'outputs' in cell:
            for output_idx, output in enumerate(cell['outputs']):
                # Check for image data
                if 'data' in output and 'image/png' in output['data']:
                    plot_count += 1

                    # Get the base64 image data
                    img_data = output['data']['image/png']

                    # Decode and save
                    img_bytes = base64.b64decode(img_data)

                    # Determine plot name based on cell content
                    cell_source = ''.join(cell.get('source', []))

                    # Try to extract a meaningful name from the cell
                    if 'plot_time_series' in cell_source:
                        plot_name = f"01_time_series_overview"
                    elif 'Chronos2 Forecast Accuracy' in cell_source or 'forecast vs actual' in cell_source:
                        plot_name = f"02_chronos_forecast_accuracy"
                    elif 'Impact of Forecast Error' in cell_source:
                        plot_name = f"03_forecast_error_impact"
                    elif 'MMM MAPE degradation' in cell_source:
                        plot_name = f"04_mape_degradation_curve"
                    else:
                        plot_name = f"plot_{plot_count:02d}_cell_{cell_idx}"

                    # Add suffix if multiple plots in same category
                    output_path = output_dir / f"{plot_name}.png"
                    counter = 1
                    while output_path.exists():
                        output_path = output_dir / f"{plot_name}_{counter}.png"
                        counter += 1

                    with open(output_path, 'wb') as f:
                        f.write(img_bytes)

                    print(f"Saved: {output_path.name}")

    print(f"\nTotal plots exported: {plot_count}")

if __name__ == "__main__":
    notebook_path = Path("notebooks/07_chronos_pymc_marketing.ipynb")
    output_dir = Path("blogs/images/chronos_mmm")

    extract_plots_from_notebook(notebook_path, output_dir)