# Media Mix Modeling Synthetic Dataset Guide

## Overview

This dataset contains synthetic data generated for Media Mix Modeling (MMM) experiments and benchmarking. The data simulates a realistic marketing scenario with multiple advertising channels, control variables, and known ground truth parameters for validation purposes.

## Available Data Files

| File | Format | Description | Size |
|------|--------|-------------|------|
| `data/mmm_data.csv` | CSV | Main time series data with all variables | 10 KB |
| `data/baseline_components.csv` | CSV | Baseline sales decomposition | 7.8 KB |
| `data/channel_contributions.csv` | CSV | Channel contributions after transformations | 8.3 KB |
| `data/ground_truth_parameters.json` | JSON | True ROAS, attribution, and transformation parameters | 1.8 KB |
| `data/dataset_summary.json` | JSON | Human-readable dataset summary | 2.3 KB |

All files use portable formats (CSV and JSON) for better interoperability with different tools and languages.

## Dataset Structure

### CSV Files

The data is split into three main CSV files for easier access:

1. **`mmm_data.csv`**: Main DataFrame with time series observations (104 rows × 11 columns)
2. **`baseline_components.csv`**: Baseline sales decomposition (104 rows × 4 columns, indexed by date and geo)
3. **`channel_contributions.csv`**: Channel contributions after transformations (104 rows × 4 columns, indexed by date and geo)

### JSON Files

1. **`ground_truth_parameters.json`**: Contains transformation parameters, ROAS values, and attribution percentages
2. **`dataset_summary.json`**: Contains dataset metadata, channel configurations, and key metrics

---

## 1. Main Data

**File**: `data/mmm_data.csv`
**Type**: CSV with header row
**Shape**: 104 rows × 11 columns
**Frequency**: Weekly data
**Date Range**: 2020-01-05 to 2021-12-26 (2 years)
**Geography**: Single region ("Local")

### Loading the Data

**Python (pandas)**:
```python
import pandas as pd
df = pd.read_csv('data/mmm_data.csv', parse_dates=['date'])
```

**R**:
```r
library(readr)
df <- read_csv('data/mmm_data.csv')
df$date <- as.Date(df$date)
```

**Julia**:
```julia
using CSV, DataFrames, Dates
df = CSV.read("data/mmm_data.csv", DataFrame)
df.date = Date.(df.date)
```

### Columns

| Column | Type | Description | Range | Mean | Std Dev |
|--------|------|-------------|-------|------|---------|
| `date` | datetime64[ns] | Week start date | 2020-01-05 to 2021-12-26 | - | - |
| `geo` | object | Geographic region identifier | "Local" | - | - |
| `x1_Search-Ads` | float64 | Search advertising spend | 0 - 654.89 | 152.11 | 141.75 |
| `x2_Social-Media` | float64 | Social media advertising spend | 0 - 654.89 | 152.11 | 141.75 |
| `x3_Local-Ads` | float64 | Local advertising spend | 0 - 500.00 | ~250 | - |
| `x4_Email` | float64 | Email marketing spend | 0 - 100.00 | ~50 | - |
| `c1` | float64 | Control variable 1 (Event) | 0 - 100 | ~50 | - |
| `c1_effect` | float64 | Effect of control variable 1 | 0 - 50 | ~1.7 | - |
| `c2` | float64 | Control variable 2 (Sale) | 0 - 100 | ~50 | - |
| `c2_effect` | float64 | Effect of control variable 2 | 0 - 73.67 | 3.41 | 10.33 |
| `y` | float64 | **Target variable (sales)** | 4723.87 - 13782.07 | 8522.81 | 2021.52 |

### Channel Characteristics

#### x1_Search-Ads (Search Advertising)
- **Pattern**: Linear trend (increasing over time)
- **Base Spend**: 100.0
- **Base Effectiveness**: 1.5
- **Spend Characteristics**: Continuous variable with gradual increase
- **Adstock**: No carryover effect (α = 0)
- **Saturation**: Hill function with slope=1, kappa=1

#### x2_Social-Media (Social Media Advertising)
- **Pattern**: Seasonal with periodic fluctuations
- **Base Spend**: 500.0
- **Base Effectiveness**: 1.2
- **Spend Characteristics**: Seasonal variation
- **Adstock**: Moderate carryover (α = 0.2, max lag = 8 weeks)
- **Saturation**: Hill function with slope=1.5, kappa=0.8

#### x3_Local-Ads (Local Advertising)
- **Pattern**: On-off (intermittent campaigns)
- **Base Spend**: 500.0
- **Base Effectiveness**: 0.9
- **Spend Characteristics**: Binary activation pattern
- **Adstock**: Strong carryover (α = 0.4, max lag = 8 weeks)
- **Saturation**: Hill function with slope=1, kappa=1.5

#### x4_Email (Email Marketing)
- **Pattern**: On-off (intermittent campaigns)
- **Base Spend**: 100.0
- **Base Effectiveness**: 1.2
- **Spend Characteristics**: Binary activation pattern
- **Adstock**: Moderate carryover (α = 0.3, max lag = 8 weeks)
- **Saturation**: Hill function with slope=2, kappa=0.5

### Control Variables

#### c1 (Event)
- **Pattern**: On-off activation
- **Base Value**: 100.0
- **Base Effectiveness**: 0.5
- **Purpose**: Captures effect of special events

#### c2 (Sale)
- **Pattern**: On-off activation
- **Base Value**: 50.0
- **Base Effectiveness**: 0.5
- **Purpose**: Captures effect of sales/promotions

---

## 2. Ground Truth Data

This section describes the true parameter values and intermediate calculations used to generate the data, enabling parameter recovery validation.

### 2.1 Ground Truth Parameters

**File**: `data/ground_truth_parameters.json`

This JSON file contains the true parameter values for validation purposes.

**Loading the Parameters**:

**Python**:
```python
import json
with open('data/ground_truth_parameters.json', 'r') as f:
    ground_truth = json.load(f)
```

**R**:
```r
library(jsonlite)
ground_truth <- fromJSON('data/ground_truth_parameters.json')
```

**Structure**:
```json
{
  "transformation_parameters": {
    "channels": {
      "Search-Ads": {
        "Local": {
          "adstock_function": "geometric_adstock",
          "adstock_params": {"alpha": 0, "l_max": 8},
          "saturation_function": "hill_function",
          "saturation_params": {"slope": 1, "kappa": 1}
        }
      }
    }
  },
  "roas_values": {...},
  "attribution_percentages": {...}
}
```

**Adstock Parameters**:
- `alpha`: Decay rate (0 = no carryover, higher = longer carryover)
- `l_max`: Maximum lag periods (8 weeks)

**Saturation Parameters**:
- `slope`: Steepness of saturation curve
- `kappa`: Half-saturation point

### 2.2 Baseline Components

**File**: `data/baseline_components.csv`

CSV file (104 rows × 4 columns) containing decomposition of baseline sales:

| Column | Description | Mean | Range |
|--------|-------------|------|-------|
| `date` | Week start date | - | 2020-01-05 to 2021-12-26 |
| `geo` | Geographic region | - | "Local" |
| `base_sales` | Base sales rate | 5070.66 | 4293.35 - 6069.41 |
| `trend` | Trend component | 0.0 | 0.0 |
| `seasonal` | Seasonal component | ~0.0 | -0.2 to 0.2 |
| `baseline_sales` | Total baseline (sum of above) | 5454.98 | 3757.14 - 7421.91 |

**Loading**:
```python
import pandas as pd
baseline = pd.read_csv('data/baseline_components.csv', parse_dates=['date'])
```

### 2.3 ROAS Values

True Return on Ad Spend (ROAS) for each channel (available in `ground_truth_parameters.json`):

| Channel | ROAS | Interpretation |
|---------|------|----------------|
| `Search-Ads` | 8.19 | Each $1 spent generates $8.19 in sales |
| `Social-Media` | 21.95 | Each $1 spent generates $21.95 in sales |
| `Local-Ads` | 3.32 | Each $1 spent generates $3.32 in sales |
| `Email` | 32.13 | Each $1 spent generates $32.13 in sales |

**Most Efficient**: Email marketing
**Least Efficient**: Local advertising

### 2.4 Attribution Percentages

Percentage of non-baseline sales attributed to each channel (available in `ground_truth_parameters.json`):

| Channel | Attribution % | Interpretation |
|---------|---------------|----------------|
| `Search-Ads` | 40.75% | Largest contributor to incremental sales |
| `Social-Media` | 14.20% | Moderate contributor |
| `Local-Ads` | 26.59% | Second largest contributor |
| `Email` | 18.46% | Moderate contributor |

**Total**: 100% (excludes baseline sales)

### 2.5 Channel Contributions

**File**: `data/channel_contributions.csv`

CSV file (104 rows × 4 columns) containing the contribution of each channel to sales after applying adstock and saturation transformations:

| Column | Description | Mean | Range |
|--------|-------------|------|-------|
| `date` | Week start date | - | 2020-01-05 to 2021-12-26 |
| `geo` | Geographic region | - | "Local" |
| `contribution_x1_Search-Ads` | Transformed contribution from search ads | 1245.78 | 0 - 3750.00 |
| `contribution_x2_Social-Media` | Transformed contribution from social media | - | - |
| `contribution_x3_Local-Ads` | Transformed contribution from local ads | - | - |
| `contribution_x4_Email` | Transformed contribution from email | 564.36 | 0 - 4921.72 |

**Loading**:
```python
import pandas as pd
contributions = pd.read_csv('data/channel_contributions.csv', parse_dates=['date'])
```

These values represent the **actual effect** of each channel's spend on sales, accounting for:
1. Adstock (carryover effects)
2. Saturation (diminishing returns)
3. Channel effectiveness

---

## 3. Dataset Summary

**File**: `data/dataset_summary.json`

This JSON file contains a human-readable summary of the dataset configuration and key metrics.

**Loading**:
```python
import json
with open('data/dataset_summary.json', 'r') as f:
    summary = json.load(f)
```

The summary contains the complete specification of how the synthetic data was generated.

### Key Attributes

| Attribute | Value | Description |
|-----------|-------|-------------|
| `n_periods` | 104 | Number of time periods (weeks) |
| `start_date` | None | Auto-generated weekly dates |
| `seed` | 333 | Random seed for reproducibility |

### Regions Configuration
- **Number of regions**: 1
- **Region name**: "Local"
- **Base sales rate**: 5000.0
- **Sales trend**: 0.0 (no overall trend)
- **Sales volatility**: 0.1
- **Seasonal amplitude**: 0.2
- **Seasonal phase**: 0.0

### Transformation Configuration
- **Adstock function**: `geometric_adstock`
- **Saturation function**: `hill_function`
- Parameters vary by channel (see `transformation_parameters`)

---

## Data Generation Process

The synthetic data was generated using the following process:

1. **Baseline Sales Generation**:
   - Base sales rate: 5000.0 per period
   - Added seasonal variation (amplitude: ±20%)
   - Added random noise (volatility: 10%)
   - No linear trend (trend = 0)

2. **Channel Spend Generation**:
   - Each channel follows its specific pattern (linear trend, seasonal, or on-off)
   - Spend values vary based on base spend and volatility parameters
   - Random activation for on-off channels

3. **Transformation Application**:
   - Adstock: Applied geometric decay to capture carryover effects
   - Saturation: Applied Hill function to model diminishing returns
   - Effectiveness: Multiplied by channel-specific effectiveness coefficient

4. **Sales Calculation**:
   ```
   y = baseline_sales +
       contribution_x1 + contribution_x2 + contribution_x3 + contribution_x4 +
       c1_effect + c2_effect
   ```

---

## Usage Notes

### Quick Start Example

**Python**:
```python
import pandas as pd
import json

# Load main data
df = pd.read_csv('data/mmm_data.csv', parse_dates=['date'])

# Load ground truth parameters
with open('data/ground_truth_parameters.json', 'r') as f:
    ground_truth = json.load(f)

# Load baseline and contributions (for validation)
baseline = pd.read_csv('data/baseline_components.csv', parse_dates=['date'])
contributions = pd.read_csv('data/channel_contributions.csv', parse_dates=['date'])

# Load dataset summary
with open('data/dataset_summary.json', 'r') as f:
    summary = json.load(f)
```

### For Model Training
- Use columns `x1_Search-Ads`, `x2_Social-Media`, `x3_Local-Ads`, `x4_Email`, `c1`, `c2` as features
- Use column `y` as the target variable (from `mmm_data.csv`)
- Include `date` for temporal modeling
- `geo` can be used for hierarchical/pooled modeling (currently single region)

### For Validation
- Compare estimated ROAS against `ground_truth_parameters.json` → `roas_values`
- Compare estimated attribution against `ground_truth_parameters.json` → `attribution_percentages`
- Compare estimated adstock/saturation parameters against `ground_truth_parameters.json` → `transformation_parameters`
- Decompose predictions and compare against `baseline_components.csv` and `channel_contributions.csv`

### For Benchmarking
- The `dataset_summary.json` contains all configuration details for reference
- Ground truth enables calculation of exact parameter recovery metrics
- Known transformations allow testing of different model specifications

---

## Key Insights from the Data

1. **Sales Distribution**: Sales range from ~4,724 to ~13,782 with a mean of ~8,523, showing substantial variation driven by marketing activities and baseline fluctuations.

2. **Channel Efficiency**: Email marketing has the highest ROAS (32.13) despite having the lowest spend, suggesting it's the most efficient channel.

3. **Channel Attribution**: Search ads contribute the most to incremental sales (40.75%), despite having a moderate ROAS (8.19), due to higher overall spend.

4. **Carryover Effects**: Social media, local ads, and email all exhibit carryover effects (α > 0), meaning their impact extends beyond the immediate period.

5. **Saturation Patterns**: All channels use Hill function saturation with varying steepness and half-saturation points, modeling realistic diminishing returns.

6. **Baseline Dominance**: Baseline sales average ~5,455, representing ~64% of total average sales (8,523), with media channels driving the remaining ~36%.

---

## Data Quality & Characteristics

- **Completeness**: No missing values
- **Consistency**: All numeric columns have appropriate data types
- **Realism**: Spend patterns and relationships mimic real-world marketing data
- **Reproducibility**: Random seed (333) ensures exact reproduction
- **Validation-Ready**: Complete ground truth enables rigorous model validation

---

## File Format Details

### CSV Files
All CSV files use:
- **Delimiter**: Comma (`,`)
- **Header**: First row contains column names
- **Date Format**: `YYYY-MM-DD` (ISO 8601)
- **Numeric Format**: Standard floating point with decimal point
- **Encoding**: UTF-8

Files with date indices (`baseline_components.csv`, `channel_contributions.csv`) include `date` and `geo` as regular columns rather than multi-index.

### JSON Files
All JSON files use:
- **Format**: Pretty-printed with 2-space indentation
- **Encoding**: UTF-8
- **Numeric Precision**: Full precision preserved for all float values

## Working with Different Languages

### Python
All files can be loaded using standard libraries (`pandas`, `json`).

### R
Use `readr` for CSV files and `jsonlite` for JSON files:
```r
library(readr)
library(jsonlite)

mmm_data <- read_csv('data/mmm_data.csv')
ground_truth <- fromJSON('data/ground_truth_parameters.json')
```

### Julia
Use `CSV.jl` and `JSON.jl`:
```julia
using CSV, DataFrames, JSON

mmm_data = CSV.read("data/mmm_data.csv", DataFrame)
ground_truth = JSON.parsefile("data/ground_truth_parameters.json")
```

### Other Languages
Standard CSV and JSON parsers will work with any language that supports these formats.

---

## References

For more details on the data generation process, see:
- `mmm_param_recovery/data_generator/core.py`: Main data generation logic
- `mmm_param_recovery/data_generator/config.py`: Configuration classes
- `mmm_param_recovery/data_generator/transforms.py`: Adstock and saturation functions

---

## File Manifest

| File | Size | Format | Description |
|------|------|--------|-------------|
| `data/mmm_data.csv` | 10 KB | CSV | Main time series data |
| `data/baseline_components.csv` | 7.8 KB | CSV | Baseline decomposition |
| `data/channel_contributions.csv` | 8.3 KB | CSV | Channel contributions |
| `data/ground_truth_parameters.json` | 1.8 KB | JSON | True parameters for validation |
| `data/dataset_summary.json` | 2.3 KB | JSON | Dataset metadata and config |

**Total Data Package**: ~30 KB
