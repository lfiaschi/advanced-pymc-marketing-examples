"""
Generate figures for the blog post from the notebook data
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
output_dir = Path("blog_figures")
output_dir.mkdir(exist_ok=True)

# Load the notebook to extract data
with open('notebooks/05_comprehensive_prior_sensitivity.ipynb', 'r') as f:
    notebook = json.load(f)

print(f"Figures will be saved to {output_dir}/")

# Figure 1: Parameter Sensitivity Ranking
fig, ax = plt.subplots(figsize=(10, 6))

parameters = ['Transformation\n(adstock, saturation)', 'Baseline\n(intercept)',
              'Effect Size\n(saturation_beta)', 'Noise\n(likelihood)', 'Control\n(gamma)']
cv_values = [1.77, 0.31, 0.25, 0.15, 0.10]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#95a5a6']

bars = ax.bar(parameters, cv_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Coefficient of Variation (%)', fontsize=12)
ax.set_title('Prior Sensitivity by Parameter Category\n(Lower is More Robust)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 2.0)

# Add value labels on bars
for bar, val in zip(bars, cv_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add interpretation bands
ax.axhspan(0, 0.5, alpha=0.1, color='green', label='Negligible sensitivity')
ax.axhspan(0.5, 1.0, alpha=0.1, color='yellow', label='Low sensitivity')
ax.axhspan(1.0, 2.0, alpha=0.1, color='orange', label='Moderate sensitivity')

ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'parameter_sensitivity_ranking.png', dpi=150, bbox_inches='tight')
print("✓ Saved: parameter_sensitivity_ranking.png")
plt.close()

# Figure 2: ROAS Error Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Simulated data based on notebook results
prior_specs = ['Loose saturation_lam', 'Tight intercept (low)', 'Tight likelihood_sigma',
               'Loose likelihood_sigma', 'Tight saturation_beta', 'Default',
               'Loose saturation_beta', 'Tight gamma_control', 'Loose gamma_control',
               'Tight adstock_alpha', 'Loose adstock_alpha', 'Tight saturation_lam']
roas_errors = [20.33, 20.67, 20.75, 20.77, 20.79, 20.81, 20.85, 20.97, 21.15, 21.44, 21.62, 21.93]

# Left panel: Error distribution
ax1.barh(range(len(prior_specs)), roas_errors, color='steelblue', alpha=0.7)
ax1.set_yticks(range(len(prior_specs)))
ax1.set_yticklabels(prior_specs, fontsize=10)
ax1.set_xlabel('ROAS Mean Absolute Percentage Error (%)', fontsize=11)
ax1.set_title('Prior Specifications Ranked by Accuracy', fontsize=12, fontweight='bold')
ax1.axvline(x=20.81, color='red', linestyle='--', label='Default priors', alpha=0.7)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Right panel: Error range visualization
ax2.hist(roas_errors, bins=10, color='coral', alpha=0.7, edgecolor='black')
ax2.set_xlabel('ROAS MAPE (%)', fontsize=11)
ax2.set_ylabel('Number of Specifications', fontsize=11)
ax2.set_title(f'Distribution of Errors\n(Range: {min(roas_errors):.1f}% - {max(roas_errors):.1f}%)',
              fontsize=12, fontweight='bold')
ax2.axvline(x=np.mean(roas_errors), color='red', linestyle='--',
            label=f'Mean: {np.mean(roas_errors):.1f}%', linewidth=2)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('ROAS Estimation Accuracy Across Prior Specifications', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'roas_error_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: roas_error_distribution.png")
plt.close()

# Figure 3: Data Learning Strength
fig, ax = plt.subplots(figsize=(10, 6))

params = ['saturation_beta', 'adstock_alpha', 'saturation_lam', 'intercept']
psi_values = [0.010, 0.089, 0.010, 0.010]
contraction = [0.30, 0.42, 0.38, 0.29]

x = np.arange(len(params))
width = 0.35

bars1 = ax.bar(x - width/2, psi_values, width, label='Prior Sensitivity Index', color='#e74c3c', alpha=0.7)
bars2 = ax.bar(x + width/2, contraction, width, label='Posterior Contraction', color='#3498db', alpha=0.7)

ax.set_xlabel('Parameter', fontsize=12)
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Evidence of Strong Data Learning\n(Lower PSI = Data Dominates, Lower Contraction = More Learning)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(params)
ax.legend()

# Add threshold lines
ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='PSI < 0.1: Strong data')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Contraction < 0.5: Good learning')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'data_learning_strength.png', dpi=150, bbox_inches='tight')
print("✓ Saved: data_learning_strength.png")
plt.close()

# Figure 4: Marketing Attribution Impact
fig, ax = plt.subplots(figsize=(10, 6))

intercept_specs = ['Low baseline\n(μ=-1)', 'Default\n(μ=0)', 'High baseline\n(μ=1)', 'Loose\n(σ=5)']
attribution_pct = [35.4, 35.2, 35.1, 35.3]
posterior_intercepts = [0.403, 0.404, 0.404, 0.403]

# Create bar plot
bars = ax.bar(intercept_specs, attribution_pct, color='teal', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add posterior intercept values as text
for i, (bar, post_int) in enumerate(zip(bars, posterior_intercepts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{attribution_pct[i]:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            f'Post: {post_int:.3f}', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

ax.set_ylabel('Marketing Attribution (%)', fontsize=12)
ax.set_title('Baseline Prior Has Minimal Effect on Marketing Attribution\n(Posteriors Converge Despite Different Priors)',
             fontsize=13, fontweight='bold')
ax.set_ylim(34, 36)
ax.grid(axis='y', alpha=0.3)

# Add reference line
ax.axhline(y=35.25, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Mean: 35.25%')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / 'marketing_attribution_impact.png', dpi=150, bbox_inches='tight')
print("✓ Saved: marketing_attribution_impact.png")
plt.close()

# Figure 5: Key Takeaway Summary
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

summary_text = """
KEY FINDINGS FROM PRIOR SENSITIVITY ANALYSIS

✓ With 104 weeks of data, MMM is remarkably robust to prior choice
  • All parameter categories show CV% < 2%
  • Prior Sensitivity Index < 0.1 (data dominates)
  • Posterior contraction < 0.5 (strong learning)

⚠ Important Context:
  • All models show 20-22% ROAS error vs ground truth
  • This reflects model structure mismatch, not prior sensitivity
  • Real-world scenarios with less data may show higher sensitivity

📊 Practical Guidance:
  1. Focus on transformation priors if you must (1.77% CV)
  2. Default priors work well with sufficient data
  3. Model structure matters more than prior choice
  4. Always validate with sensitivity analysis

🎯 Bottom Line:
More data > Better priors
Model validation > Prior optimization
"""

ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3),
        fontfamily='monospace')

plt.tight_layout()
plt.savefig(output_dir / 'key_takeaways.png', dpi=150, bbox_inches='tight')
print("✓ Saved: key_takeaways.png")
plt.close()

print(f"\n✅ All figures generated successfully in {output_dir}/")