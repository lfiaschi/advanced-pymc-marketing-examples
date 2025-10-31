# Stop Agonizing Over MMM Priors: A Data-Driven Analysis and Playbook to Set Them

You’ve been there—three hours deep into debugging why your Media Mix Model won’t converge, another hour tweaking priors, and your stakeholders asking why the model recommends spending $0 on your top-performing channel. Meanwhile, your posterior plots look like abstract art instead of inference.

Here’s the truth: priors in MMM aren’t just statistical formalities—they’re the backbone of model stability, interpretability, and business credibility. But which priors actually matter? Where should you invest effort? And how can you set them in a principled, data-driven way?

To find out, we ran a large-scale sensitivity analysis across 14 prior configurations and six parameter families on a realistic synthetic datasets with known ground truth. The results challenge common intuition: transformation parameters dominate, effect sizes come next, and several priors you’ve been agonizing over barely move the needle.

## The Prior Problem No One Talks About

Media Mix Models are inherently hard to identify. You’re trying to disentangle correlated marketing channels, capture delayed effects beyond the observation window, and quantify saturation you can’t directly observe. In such a high-dimensional, noisy system, priors serve three purposes:

1. **Computational tractability**: Wrong priors = hours of wasted compute on models that won't converge
2. **Parameter identifiability**: Distinguishing signal from noise when channels are correlated
3. **Business alignment**: Encoding domain knowledge about marketing effectiveness

Yet most practitioners either rely on defaults or tweak priors blindly until the model “behaves.” The result is often fragile inference and poor reproducibility.

## Domain Knowledge Should Guide Priors (When Available)

Ideally, priors should reflect domain knowledge—for example, that brand campaigns have longer carry-over or that search ads saturate quickly. In practice, though, this information is often incomplete. When that happens, you should at least be aware which priors matter most so you can interpret results responsibly.

This is not an argument for “prior tuning” to optimize predictive accuracy. Over-adjusting priors risks hiding model misspecification under the illusion of better fit. Instead, the goal is to understand where prior assumptions exert influence and where the data can safely speak for itself.

## How We Tested Prior Sensitivity

We used 104 weeks of synthetic data with known ground truth and systematically varied priors for each key component of a typical MMM


```python
# Core parameters we tested
parameter_categories = {
    "transformation": ["adstock_alpha", "saturation_lam"],
    "effect_size": ["saturation_beta"],
    "baseline": ["intercept"],
    "noise": ["likelihood_sigma"],
    "control": ["gamma_control"]
}

# For each, we tested tight vs. loose specifications
prior_specs = {
    "tight_saturation_beta": Prior("HalfNormal", sigma=0.5),  # Conservative
    "default": None,  # PyMC-Marketing defaults
    "loose_saturation_beta": Prior("HalfNormal", sigma=5),   # Agnostic
    # ... and 11 more specifications
}
```

For each parameter, we compared tight, default, and loose priors and quantified sensitivity using:

- Coefficient of Variation (CV%) – relative variation in ROAS across prior settings.
- Prior Sensitivity Index (PSI) – between-prior variance divided by within-posterior variance.
- Posterior Contraction – how much data reduces prior uncertainty.

## What We Found: The Hierarchy of Prior Importance

Our analysis revealed a clear hierarchy of which priors deserve attention:

![Parameter Sensitivity Ranking](../blog_figures/parameter_sensitivity_ranking.png)

### Finding 1: Transformation Parameters Drive Most of the Variance

Adstock and saturation priors were 7× more influential than effect sizes:

- **Adstock (carryover)** and **saturation point** parameters had CV% of 1.77%
- Effect size (saturation_beta) showed only 0.25% CV%
- Baseline (intercept) was at 0.31% CV%

While 1.77% might seem low in absolute terms, consider this: for a company spending $10M annually on marketing, a 2% shift in ROAS estimates translates to $200,000 in potentially misallocated budget. More critically, transformation parameters directly affect:

1. **Convergence speed**: Wrong adstock priors can 10x your sampling time
2. **Channel attribution**: High carryover attributes current sales to past spend
3. **Optimization recommendations**: Saturation points determine where to cut spend

### Finding 2: The 20 % ROAS Error That Priors Can’t Fix

Every configuration produced ~20 % ROAS error versus ground truth. The culprit wasn’t the priors—it was the model structure. The ground truth used a Hill saturation curve; our model used logistic. No amount of prior tweaking could remove that structural bias.

The lesson is that if your functional form doesn’t match business dynamics, better priors won’t save you.

![ROAS Error Distribution](../blog_figures/roas_error_distribution.png)

### Finding 3: Strong Data Doesn't Eliminate Prior Importance

With 104 clean weeks, data dominated (PSI < 0.1). But on real data noise, collinearity, and shifting effects make priors far more influential. With 52 weeks or correlated channels, transformation priors can determine whether the model converges at all.

![Data Learning Strength](../blog_figures/data_learning_strength.png)

- Prior Sensitivity Index < 0.1 (data dominates)
- Posterior contraction < 0.5 (variance reduced by >50%)
- Convergence achieved for all specifications

In practice, with limited and imperfect data, prior choice becomes exponentially more important.

## A Practical Playbook for Setting MMM Priors

Based on our analysis, here's your systematic approach to prior specification:

### Priority 1: Transformation Parameters (High Impact)

**Adstock (Carryover) Alpha**
```python
# Start with your business context
if brand_focused:
    # Brand advertising has long memory
    adstock_alpha = Prior("Beta", alpha=5, beta=2)  # Mean ~0.7
elif direct_response:
    # Performance marketing decays quickly
    adstock_alpha = Prior("Beta", alpha=3, beta=7)  # Mean ~0.3
else:
    # Uncertain? Use weakly informative
    adstock_alpha = Prior("Beta", alpha=1, beta=3)  # Mean ~0.25
```

**Why this matters**: Wrong adstock priors can cause:
- Convergence failures (divergences, low ESS)
- Attribution errors (crediting wrong time periods)
- Poor out-of-sample predictions

**Saturation Lambda (Diminishing Returns)**
```python
# Based on historical spend ranges
spend_range = X[channel].max() - X[channel].min()
typical_spend = X[channel].median()

if expect_early_saturation:
    # Diminishing returns kick in quickly
    saturation_lam = Prior("Gamma", alpha=5, beta=2/typical_spend)
else:
    # Linear effects persist longer
    saturation_lam = Prior("Gamma", alpha=2, beta=0.5/typical_spend)
```

### Priority 2: Effect Size Parameters (Medium Impact)

**Saturation Beta (Channel Effectiveness)**
```python
# Use business priors about channel ROI
if established_channel:
    saturation_beta = Prior("HalfNormal", sigma=2)  # Default
elif experimental_channel:
    saturation_beta = Prior("HalfNormal", sigma=0.5)  # Conservative
elif high_confidence_channel:
    saturation_beta = Prior("HalfNormal", sigma=5)  # Allow large effects
```

### Priority 3: Baseline and Noise (Low Impact)

These showed minimal sensitivity in our analysis, suggesting defaults work well:

```python
# Usually safe to use defaults
intercept = Prior("Normal", mu=0, sigma=2)
likelihood_sigma = Prior("HalfNormal", sigma=1)
gamma_control = Prior("Normal", mu=0, sigma=2)
```

## The Marketing Attribution Paradox

One surprising finding deserves special attention. Despite varying the intercept prior from "mostly organic sales" to "mostly marketing-driven," the marketing attribution percentage barely budged:

![Marketing Attribution Impact](../blog_figures/marketing_attribution_impact.png)

All specifications attributed ~35% of sales to marketing, with posteriors converging to intercept ≈ 0.404 regardless of prior. This demonstrates that with sufficient data, the model correctly identifies the baseline/marketing split—but only if your model structure is correct.

## Validation Protocol: Trust but Verify

Never deploy an MMM without systematic validation. Here's our recommended protocol:

### Step 1: Prior Predictive Checks
```python
# Before fitting, simulate from priors
mmm.sample_prior_predictive(samples=1000)

# Check if simulated data looks reasonable
# - Are ROAS values in plausible ranges?
# - Do saturation curves make business sense?
# - Are carryover patterns realistic?
```

### Step 2: Sensitivity Analysis
Run at least 3 specifications:
1. Conservative (tight priors based on pessimistic assumptions)
2. Default (PyMC-Marketing standards)
3. Optimistic (loose priors allowing large effects)

Compare:
- ROAS estimates (should be within 20% of each other)
- Convergence diagnostics (R-hat < 1.01, ESS > 400)
- Out-of-sample predictions (use last 15% of the weeks as holdout)

This allow identifying which priors affect the covergence and the accuracy of your models.

### Step 3: Posterior Predictive Checks
```python
# After fitting, validate model behavior
mmm.sample_posterior_predictive()
```
Check for:
- Prediction intervals covering actual data
- No systematic over/under prediction
- Reasonable accuracy measured with CRPS, R2, RMSE, Durbin-Watson

## When Prior Choice Becomes Critical

Our analysis identified specific scenarios where prior specification can make or break your MMM:

### Red Flag Scenarios

1. **Limited Data** (< 104 weeks)
   - Prior influence increases exponentially
   - Focus on transformation parameters
   - Consider informative priors from previous campaigns

2. **High Multicollinearity** (correlation > 0.7 between channels)
   - Priors help identify individual channel effects
   - Critical for channels that move together (e.g., TV + Radio)

3. **New Channels** (< 13 weeks of history)
   - Data can't distinguish saturation from linear effects
   - Business knowledge becomes essential

4. **Computational Constraints**
   - Wrong transformation priors = 10x longer sampling

## Focus as Much on Structure as on Priors

Our 20 % systematic error came from model mismatch, not from prior misspecification. The takeaway:

- Verify that your saturation and adstock functions match real market dynamics.
- Consider time-varying or delayed effects where appropriate.
- Use priors to stabilize inference and encode business assumptions, not to improve accuracy

The good news is that these model specifications can be tuned with other model selection techniques [BLOG POST LINK TO OPTUNA]

## Practical Recommendations

Based on our comprehensive analysis, here's your action plan:

### For Practitioners

1. **Start with transformation parameters**
   - These have highest impact on both estimates and convergence
   - Use business knowledge about carryover and saturation
   - Validate with sensitivity analysis

2. **Don't overthink effect size priors**
   - Data learns these well with 50+ weeks
   - Defaults usually work fine
   - Focus on getting the direction right (positive effects)

3. **Always run sensitivity analysis**
   - It's not optional—it's quality control
   - Document which priors were tested
   - Report the range of estimates to stakeholders

## The Path Forward

Priors don’t make or break your MMM alone—model design and data quality do. But setting them systematically can be the difference between robust insight and wasted computation.

Remember:

- Use business knowledge where possible.
- Know which priors matter most expecially when that knowledge is missing.
- Validate, document, and communicate impact to priors and assumptions transparently.

Stop agonizing over every prior. Focus on the ones that matter, test them rigorously, and let your data—and your domain expertise—work together.
