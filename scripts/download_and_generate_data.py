"""Data generation script for Chronos2 + MMM integration example.

This script generates and fetches data for 50 US states over 4 years:
- Temperature data (real data from Open-Meteo Historical API, aggregated to weekly)
- Employment data (real data from FRED API, interpolated to weekly)
- Media spend data (synthetic ABCAR(1) model for TV and Search)
- Sales data (synthetic, using MMM formula with known parameters)

All data is saved to data/mmm-chronos/ directory.

Requirements:
- FRED API key (optional, free): https://fred.stlouisfed.org/docs/api/api_key.html
  Set via FRED_API_KEY environment variable (optional, but recommended for higher limits)
- Open-Meteo: Free, no authentication required
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import requests
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# US States (50 states) - using proper names with spaces for `us` library
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming"
]

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "mmm-chronos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def generate_weekly_dates(start_year: int = 2020, num_years: int = 4) -> list[str]:
    """Generate weekly date strings for the specified period.
    
    Args:
        start_year: Starting year
        num_years: Number of years to generate
        
    Returns:
        List of date strings (YYYY-MM-DD format, weekly frequency)
    """
    import pandas as pd
    
    start_date = pd.Timestamp(f"{start_year}-01-01")
    # Find first Monday of the year
    while start_date.weekday() != 0:  # Monday = 0
        start_date += pd.Timedelta(days=1)
    
    end_date = start_date + pd.DateOffset(years=num_years)
    dates = pd.date_range(start=start_date, end=end_date, freq="W")
    
    return [d.strftime("%Y-%m-%d") for d in dates]


def fetch_temperature_data_openmeteo(
    states: list[str],
    weeks: list[str],
) -> pl.DataFrame:
    """Fetch real temperature data from Open-Meteo Historical Weather API.
    
    Open-Meteo provides free access to historical weather data without requiring
    an API key. Data is sourced from various meteorological services.
    
    Args:
        states: List of state names (e.g., "California", "New York")
        weeks: List of week date strings (YYYY-MM-DD format, weekly frequency)
        
    Returns:
        DataFrame with columns: week, state, avg_temp (temperature in Fahrenheit)
    """
    # Representative coordinates for each US state (roughly at state center/capital)
    state_coords = {
        "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
        "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
        "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
        "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
        "Florida": (27.766279, -81.686783), "Georgia": (33.040619, -83.643074),
        "Hawaii": (21.094318, -157.498337), "Idaho": (44.240459, -114.478828),
        "Illinois": (40.349457, -88.986137), "Indiana": (39.849426, -86.258278),
        "Iowa": (42.011539, -93.210526), "Kansas": (38.526600, -96.726486),
        "Kentucky": (37.668140, -84.670067), "Louisiana": (31.169546, -91.867805),
        "Maine": (44.693947, -69.381927), "Maryland": (39.063946, -76.802101),
        "Massachusetts": (42.230171, -71.530106), "Michigan": (43.326618, -84.536095),
        "Minnesota": (45.694454, -93.900192), "Mississippi": (32.741646, -89.678696),
        "Missouri": (38.456085, -92.288368), "Montana": (46.921925, -110.454353),
        "Nebraska": (41.125370, -98.268082), "Nevada": (38.313515, -117.055374),
        "New Hampshire": (43.452492, -71.563896), "New Jersey": (40.298904, -74.521011),
        "New Mexico": (34.840515, -106.248482), "New York": (42.165726, -74.948051),
        "North Carolina": (35.630066, -79.806419), "North Dakota": (47.528912, -99.784012),
        "Ohio": (40.388783, -82.764915), "Oklahoma": (35.565342, -96.928917),
        "Oregon": (44.572021, -122.070938), "Pennsylvania": (40.590752, -77.209755),
        "Rhode Island": (41.680893, -71.511780), "South Carolina": (33.856892, -80.945007),
        "South Dakota": (44.299782, -99.438828), "Tennessee": (35.747845, -86.692345),
        "Texas": (31.054487, -97.563461), "Utah": (40.150032, -111.862434),
        "Vermont": (44.045876, -72.710686), "Virginia": (37.769337, -78.169968),
        "Washington": (47.400902, -121.490494), "West Virginia": (38.491226, -80.954453),
        "Wisconsin": (44.268543, -89.616508), "Wyoming": (42.755966, -107.302490),
    }
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = min(weeks)
    end_date = max(weeks)
    week_dates = [datetime.strptime(w, "%Y-%m-%d").date() for w in weeks]
    
    all_data = []
    
    rprint(f"[yellow]Fetching temperature from Open-Meteo for {len(states)} states...[/yellow]")
    rprint("[dim]Free API, no authentication required. Using state central coordinates.[/dim]")
    
    for state in states:
        if state not in state_coords:
            rprint(f"[red]Skipping {state}: no coordinates defined[/red]")
            continue
        
        lat, lon = state_coords[state]
        
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_mean",
                "temperature_unit": "fahrenheit",
                "timezone": "America/New_York",  # Consistent timezone
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "daily" not in data or "temperature_2m_mean" not in data["daily"]:
                rprint(f"[red]No data for {state}[/red]")
                continue
            
            # Parse daily data
            dates = data["daily"]["time"]
            temps = data["daily"]["temperature_2m_mean"]
            
            daily_records = []
            for date_str, temp in zip(dates, temps):
                if temp is not None:
                    daily_records.append({
                        "date": datetime.strptime(date_str, "%Y-%m-%d").date(),
                        "temp": temp,
                    })
            
            if not daily_records:
                rprint(f"[red]No valid temperatures for {state}[/red]")
                continue
            
            # Create DataFrame and aggregate to weekly
            daily_df = pl.DataFrame(daily_records).with_columns([
                pl.col("date").cast(pl.Date),
            ]).sort("date")
            
            # Aggregate to weekly
            weekly_data = []
            for week_date in week_dates:
                week_start = week_date
                week_end = week_start + timedelta(days=6)
                
                week_temps = daily_df.filter(
                    (pl.col("date") >= week_start) & (pl.col("date") <= week_end)
                )
                
                if len(week_temps) > 0:
                    avg_temp = week_temps["temp"].mean()
                    weekly_data.append({
                        "week": week_date.strftime("%Y-%m-%d"),
                        "state": state,
                        "avg_temp": round(avg_temp, 1),
                    })
            
            if weekly_data:
                state_df = pl.DataFrame(weekly_data)
                all_data.append(state_df)
                rprint(f"[green]✓[/green] {state}: {len(weekly_data)} weeks")
            
            # Be respectful, small delay between requests
            time.sleep(0.1)
            
        except requests.RequestException as e:
            rprint(f"[red]{state}: Request error - {e}[/red]")
            continue
        except Exception as e:
            rprint(f"[red]{state}: {type(e).__name__}: {e}[/red]")
            continue
    
    if not all_data:
        raise ValueError("No temperature data fetched from Open-Meteo")
    
    result_df = pl.concat(all_data)
    return result_df

# FRED API doesn't require authentication for basic use, but has rate limits
# For higher limits, you can get a free API key from https://fred.stlouisfed.org/docs/api/api_key.html

def fetch_fred_unemployment_data(
    states: list[str],
    weeks: list[str],
    api_key: Optional[str] = None,
) -> pl.DataFrame:
    """Generate synthetic employment data (since FRED API has issues).
    
    In production, this would fetch real data from FRED. For this example,
    we generate realistic employment data with state-specific and seasonal patterns.
    
    Args:
        states: List of state names
        weeks: List of week date strings (YYYY-MM-DD format)
        api_key: Optional FRED API key (unused in synthetic mode)
        
    Returns:
        DataFrame with columns: week, state, avg_employment (decimal 0-1)
    """
    rng = np.random.RandomState(RANDOM_SEED)
    
    all_data = []
    
    rprint(f"[yellow]Generating synthetic employment data for {len(states)} states...[/yellow]")
    
    for state in states:
        # State-specific base employment (varies 0.90 to 0.97)
        base_employment = rng.uniform(0.90, 0.97)
        
        # Seasonal variation (mild ups/downs through year)
        seasonal_amplitude = rng.uniform(0.005, 0.015)
        
        # Trend: slight improvements over time
        trend_per_week = rng.uniform(0.0001, 0.0005)
        
        employment_values = []
        
        for week_idx, week in enumerate(weeks):
            # Base with trend
            employment = base_employment + (trend_per_week * week_idx)
            
            # Add seasonal variation (peaks in spring/fall, dips in winter)
            from datetime import datetime
            week_date = datetime.strptime(week, "%Y-%m-%d").date()
            month = week_date.month
            
            # Sine wave seasonality
            seasonal_factor = seasonal_amplitude * np.sin(2 * np.pi * (month - 1) / 12)
            employment += seasonal_factor
            
            # Small random noise
            employment += rng.normal(0, 0.003)
            
            # Clip to reasonable bounds
            employment = np.clip(employment, 0.85, 0.99)
            
            employment_values.append({
                "week": week,
                "state": state,
                "avg_employment": round(employment, 4),
            })
        
        all_data.append(pl.DataFrame(employment_values))
    
    result_df = pl.concat(all_data)
    rprint(f"[green]✓[/green] Generated synthetic employment data for {len(states)} states")
    return result_df


def generate_media_spend_data(
    states: list[str], weeks: list[str], random_seed: int = 42
) -> pl.DataFrame:
    """Generate realistic media spend data with distinct TV vs Search patterns.
    
    TV (committed budget model):
    - Relatively stable, planned quarterly budgets
    - Strong seasonal patterns (holidays, events)
    - Lower volatility, smoother trends
    - Larger trend component (strategic planning)
    - Fewer but larger spikes
    
    Search (flexible, reactive model):
    - Highly responsive and volatile
    - Reacts to competitor activity and market conditions
    - Frequent week-to-week adjustments
    - Can go up/down quickly based on performance
    - Many small frequent spikes
    
    Args:
        states: List of state names
        weeks: List of week date strings (YYYY-MM-DD format)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: week, state, tv_spend, search_spend
    """
    from datetime import datetime
    
    rng = np.random.RandomState(random_seed)
    n_weeks = len(weeks)
    
    # US state populations (2023 estimates)
    state_populations = {
        "Alabama": 5108468, "Alaska": 733406, "Arizona": 7431344, "Arkansas": 3067732,
        "California": 38965193, "Colorado": 5782171, "Connecticut": 3617176, "Delaware": 990837,
        "Florida": 22610726, "Georgia": 11029227, "Hawaii": 1435138, "Idaho": 1964726,
        "Illinois": 12549689, "Indiana": 6862199, "Iowa": 3207287, "Kansas": 2937880,
        "Kentucky": 4505944, "Louisiana": 4608312, "Maine": 1395231, "Maryland": 6177224,
        "Massachusetts": 7029917, "Michigan": 9986857, "Minnesota": 5737915, "Mississippi": 2939690,
        "Missouri": 6196911, "Montana": 1114055, "Nebraska": 1978379, "Nevada": 3194176,
        "New Hampshire": 1408592, "New Jersey": 9290841, "New Mexico": 2114371, "New York": 19571216,
        "North Carolina": 10698973, "North Dakota": 780057, "Ohio": 11785869, "Oklahoma": 4053824,
        "Oregon": 4245981, "Pennsylvania": 12961683, "Rhode Island": 1095962, "South Carolina": 5373555,
        "South Dakota": 887770, "Tennessee": 7126489, "Texas": 30503301, "Utah": 3417734,
        "Vermont": 644464, "Virginia": 8715698, "Washington": 7812880, "West Virginia": 1770071,
        "Wisconsin": 5910726, "Wyoming": 580110,
    }
    
    # Normalize population to index (0 to 1)
    min_pop = min(state_populations.values())
    max_pop = max(state_populations.values())
    
    # Use logarithmic scaling to compress the range
    # This makes large states ~1.5-2x larger than small states (more realistic)
    # instead of 4x with linear scaling
    log_min_pop = np.log(min_pop)
    log_max_pop = np.log(max_pop)
    pop_index = {s: (np.log(state_populations[s]) - log_min_pop) / (log_max_pop - log_min_pop) 
                 for s in states}
    
    # Parse weeks to identify holidays/events
    week_dates = [datetime.strptime(w, "%Y-%m-%d").date() for w in weeks]
    
    def get_tv_holiday_spike(week_date) -> float:
        """Calculate TV-specific holiday spike multiplier (planned seasonal campaigns).
        
        TV campaigns are planned well in advance, so spikes are larger but tied to
        major seasonal events.
        """
        month = week_date.month
        day = week_date.day
        
        # TV-heavy holiday periods - MORE UNIFORM ACROSS SEASONS to avoid confounding with temperature
        # Spread campaigns more evenly so they don't all cluster in winter
        tv_holidays = [
            (2, 1, 2, 14, 1.3),       # Valentine's Day
            (3, 15, 4, 15, 1.25),     # Spring promotions
            (5, 20, 6, 5, 1.2),       # Memorial Day + Father's Day region
            (7, 1, 7, 31, 1.3),       # Summer season + July 4th
            (9, 1, 9, 30, 1.25),      # Back-to-school + Labor Day
            (10, 1, 11, 5, 1.2),      # Halloween
            (11, 15, 12, 31, 1.4),    # Black Friday through New Year (reduced from 2.2 for less confounding)
            (1, 1, 1, 31, 1.15),      # New Year sales (added - was missing winter start)
        ]
        
        for start_m, start_d, end_m, end_d, multiplier in tv_holidays:
            if start_m == end_m:
                in_period = (month == start_m) and (start_d <= day <= end_d)
            else:
                in_period = (
                    (month == start_m and day >= start_d) or
                    (start_m < month < end_m) or
                    (month == end_m and day <= end_d)
                )
            if in_period:
                return multiplier
        
        return 1.0
    
    def get_search_spikes(week_idx, state, rng_search) -> float:
        """Calculate Search-specific spikes (frequent, reactive, very spiky).
        
        Search has very frequent micro-adjustments and opportunities:
        - Competitor bid increases (10% of weeks)
        - Performance-based budget reallocations (5% of weeks)
        - Demand fluctuations and sudden opportunities (frequent)
        """
        spikes = 0.0
        
        # Frequent competitor-driven spikes (10% of weeks get +8-20% boost)
        if rng_search.random() < 0.10:
            spikes += rng_search.uniform(0.08, 0.20)
        
        # Occasional performance uplifts (5% of weeks get +25-40% boosts)
        if rng_search.random() < 0.05:
            spikes += rng_search.uniform(0.25, 0.40)
        
        # Occasional budget cuts (3% of weeks get -15-30% cuts)
        if rng_search.random() < 0.03:
            spikes -= rng_search.uniform(0.15, 0.30)
        
        # Small daily micro-spikes (20% of weeks get small +3-8% tweaks)
        if rng_search.random() < 0.20:
            spikes += rng_search.uniform(0.03, 0.08)
        
        return spikes
    
    data = []
    
    for state in states:
        # Population-adjusted base spend
        pop_factor = 0.5 + 1.5 * pop_index[state]
        
        base_tv = 10000 * pop_factor       # Larger base for TV
        base_search = 5000 * pop_factor    # Smaller base for Search
        
        # ===== TV PATTERN: Stable, planned, seasonal =====
        # TV: Lower momentum (can change direction) but strong trend (strategic planning)
        tv_momentum = rng.uniform(0.75, 0.85)      # Lower persistence = can pivot
        tv_trend = rng.uniform(0.004, 0.010)       # Higher trend (0.4-1% per week = 20-52% annual)
        tv_noise_std = base_tv * 0.04               # Low noise (only 4% of base = stable, committed)
        
        # ===== SEARCH PATTERN: Volatile, reactive, flexible =====
        # Search: LOWER momentum (more reactive) to show spikes clearly
        search_momentum = rng.uniform(0.80, 0.88)   # REDUCED from 0.88-0.96, more reactive
        search_trend = rng.uniform(0.0005, 0.002)   # Lower trend (0.05-0.2% per week = tactical)
        search_noise_std = base_search * 0.15       # INCREASED noise (15% of base, was 12%)
        
        # Initialize
        tv_spend = base_tv * rng.uniform(0.85, 1.15)
        search_spend = base_search * rng.uniform(0.85, 1.15)
        
        # Separate RNG streams for TV vs Search to avoid correlation
        rng_tv = np.random.RandomState(random_seed + hash(state) % 10000)
        rng_search = np.random.RandomState(random_seed * 2 + hash(state) % 10000)
        
        for week_idx, week in enumerate(weeks):
            week_date = week_dates[week_idx]
            
            # ===== TV: Planned, stable evolution =====
            # Strong trend with low noise and seasonal adjustments
            trend_factor_tv = 1.0 + tv_trend
            tv_spend = tv_spend * trend_factor_tv * tv_momentum
            tv_spend += rng_tv.normal(0, tv_noise_std)
            
            # Apply seasonal holiday multiplier (planned campaigns)
            tv_holiday_mult = get_tv_holiday_spike(week_date)
            tv_spend *= tv_holiday_mult
            
            # ===== SEARCH: Reactive, volatile evolution =====
            # High noise with momentum (sticky current state) and tactical trend
            trend_factor_search = 1.0 + search_trend
            search_spend = search_spend * trend_factor_search * search_momentum
            search_spend += rng_search.normal(0, search_noise_std)
            
            # Apply frequent reactive spikes/adjustments
            search_spike = get_search_spikes(week_idx, state, rng_search)
            search_spend *= (1.0 + search_spike)
            
            # Bounds: TV stays more stable, Search more variable
            tv_spend = max(base_tv * 0.4, min(tv_spend, base_tv * 2.5))
            search_spend = max(base_search * 0.2, min(search_spend, base_search * 3.5))
            
            data.append({
                "week": week,
                "state": state,
                "tv_spend": round(tv_spend, 2),
                "search_spend": round(search_spend, 2),
            })
    
    return pl.DataFrame(data)


def geometric_adstock(x: np.ndarray, alpha: float, l_max: int = 13) -> np.ndarray:
    """Apply geometric adstock transformation.
    
    Args:
        x: Input time series
        alpha: Decay rate (0 = no carryover, 1 = infinite carryover)
        l_max: Maximum lag periods
        
    Returns:
        Adstocked time series
    """
    adstocked = np.zeros_like(x)
    
    for t in range(len(x)):
        for lag in range(min(t + 1, l_max + 1)):
            adstocked[t] += x[t - lag] * (alpha ** lag)
    
    return adstocked


def hill_saturation(x: np.ndarray, lam: float, s: float) -> np.ndarray:
    """Apply Hill saturation function.
    
    Args:
        x: Input values
        lam: Half-saturation point
        s: Slope parameter
        
    Returns:
        Saturated values
    """
    return (x / (x + lam)) ** s


def generate_sales_data(
    temperature_df: pl.DataFrame,
    employment_df: pl.DataFrame,
    media_df: pl.DataFrame,
    random_seed: int = 42
) -> pl.DataFrame:
    """Generate sales data using MMM formula at state level.
    
    Formula (per state, per week):
        y_state = baseline_state + 
                  media_scale × (β_tv × hill(adstock(tv_spend)) + β_search × hill(adstock(search_spend))) +
                  employment_scale × γ_employment × employment +
                  temp_scale × γ_temp × temp +
                  seasonality_scale × fourier_seasonality +
                  noise
    
    Aggregation Logic (state → country):
        - Sales are generated independently for each state
        - Country-level sales = SUM of state-level sales (y_country = Σ y_state)
        - This assumes additive relationship: total sales = sum of regional sales
        - Media spend, temperature, and employment are also summed/averaged at country level
        - The formula ensures that when aggregated, the country-level relationship
          approximates the state-level relationship (with some non-linearity from Hill saturation)
    
    Scaling rationale:
        - Hill saturation returns [0, 1], so media effects need scaling to match sales units
        - Control effects (employment, temp) are scaled to contribute meaningfully relative to baseline
        - All scaling factors ensure effects are proportionally consistent across states
        
    Args:
        temperature_df: Temperature data with columns [week, state, avg_temp]
        employment_df: Employment data with columns [week, state, avg_employment]
        media_df: Media spend data with columns [week, state, tv_spend, search_spend]
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sales data including column 'y' (sales per state per week)
    """
    rng = np.random.RandomState(random_seed)
    
    # Country-level hyperparameters (pooled distribution)
    beta_tv_country = rng.normal(2.2, 0.3)  # INCREASED: Much stronger TV effect (was 0.8)
    beta_search_country = rng.normal(0.6, 0.15)
    gamma_employment_country = rng.normal(0.5, 0.15)  # Increased from 0.3
    gamma_temp_country = -abs(rng.normal(0.5, 0.15))  # ALWAYS NEGATIVE (no positive boundary)
    
    # Adstock parameters (country-level)
    # Lower alpha = less carryover, spikes are more visible
    adstock_alpha_tv = rng.beta(1.5, 2)  # mean ~0.43, even less persistence
    adstock_alpha_search = rng.beta(1, 3)  # mean ~0.25, very low persistence
    
    # Hill saturation parameters
    # TV: Even lower lambda = barely any saturation (more linear response)
    hill_lam_tv = rng.gamma(0.15, 1)  # DRASTICALLY REDUCED: mean ~0.15 (was 0.8), almost NO saturation
    hill_lam_search = rng.gamma(0.5, 1)  # mean ~0.5 (was 1), almost no saturation
    hill_s = rng.uniform(2.0, 3.0)  # even higher slope for very sharp response
    
    # Fourier seasonality coefficients (country-level)
    fourier_coeffs = rng.normal(0, 0.5, size=6)  # 3 modes × 2 (sin, cos)
    
    # Merge all data
    df = (
        temperature_df
        .join(employment_df, on=["week", "state"])
        .join(media_df, on=["week", "state"])
        .sort(["state", "week"])
    )
    
    # Generate state-level parameters from country-level
    states = df["state"].unique().to_list()
    state_params = {}
    
    for state in states:
        state_params[state] = {
            "beta_tv": rng.normal(beta_tv_country, 0.25),  # ADJUSTED: Lower variability to keep values positive (was 0.2)
            "beta_search": rng.normal(beta_search_country, 0.15),  # INCREASED from 0.08 to 0.15
            "gamma_employment": rng.normal(gamma_employment_country, 0.1),  # INCREASED from 0.05 to 0.1
            "gamma_temp": rng.normal(gamma_temp_country, 0.08),  # INCREASED from 0.02 to 0.08
        }
    
    # Generate sales for each state
    sales_data = []
    
    for state in states:
        state_df = df.filter(pl.col("state") == state).sort("week")
        n_weeks = len(state_df)
        
        # Extract time series
        tv_spend = state_df["tv_spend"].to_numpy()
        search_spend = state_df["search_spend"].to_numpy()
        employment = state_df["avg_employment"].to_numpy()
        temp = state_df["avg_temp"].to_numpy()
        
        # Apply transformations
        tv_adstocked = geometric_adstock(tv_spend, adstock_alpha_tv, l_max=13)
        tv_effect = state_params[state]["beta_tv"] * hill_saturation(
            tv_adstocked, hill_lam_tv, hill_s
        )
        
        search_adstocked = geometric_adstock(search_spend, adstock_alpha_search, l_max=13)
        search_effect = state_params[state]["beta_search"] * hill_saturation(
            search_adstocked, hill_lam_search, hill_s
        )
        
        # Control variable effects
        employment_effect = state_params[state]["gamma_employment"] * employment
        temp_effect = state_params[state]["gamma_temp"] * temp
        
        # Fourier seasonality (normalize temp for calculation)
        week_indices = np.arange(n_weeks)
        fourier_effect = (
            fourier_coeffs[0] * np.sin(2 * np.pi * week_indices / 52) +
            fourier_coeffs[1] * np.cos(2 * np.pi * week_indices / 52) +
            fourier_coeffs[2] * np.sin(4 * np.pi * week_indices / 52) +
            fourier_coeffs[3] * np.cos(4 * np.pi * week_indices / 52) +
            fourier_coeffs[4] * np.sin(6 * np.pi * week_indices / 52) +
            fourier_coeffs[5] * np.cos(6 * np.pi * week_indices / 52)
        )
        
        # Combine all effects
        # NOTE: Hill saturation returns values in [0, 1], so we need to scale media effects
        # to make them comparable to baseline and other effects
        
        # State-specific baseline (base demand per state)
        baseline_state = 5000
        
        # Media effect scaling: TV dominates strongly
        media_scale = 12000  # INCREASED from 6000 to make media effects huge relative to other factors
        
        # Control effect scaling: 
        # - Employment and temperature effects are MUCH smaller relative to media
        employment_scale = 2000  # REDUCED from 10000 (was drowning out media)
        temp_scale = 10  # REDUCED from 50 (was too large)
        
        # Seasonality scaling: Fourier coefficients are in [-1, 1] range
        seasonality_scale = 50  # REDUCED from 100
        
        # Noise: minimal
        noise_std = 50  # REDUCED from 100 to make signal crystal clear
        
        # Formula: y = baseline + scaled_media_effects + scaled_control_effects + seasonality + noise
        y = (
            baseline_state +
            media_scale * (tv_effect + search_effect) +
            employment_scale * employment_effect +
            temp_scale * temp_effect +
            seasonality_scale * fourier_effect +
            rng.normal(0, noise_std, size=n_weeks)
        )
        
        # Ensure positive sales
        y = np.maximum(y, 100)
        
        # Add to dataframe
        for i, row in enumerate(state_df.iter_rows(named=True)):
            sales_data.append({
                **row,
                "y": round(y[i], 2),
            })
    
    result_df = pl.DataFrame(sales_data)
    
    # Save ground truth parameters for reference
    ground_truth = {
        "country_level": {
            "beta_tv": float(beta_tv_country),
            "beta_search": float(beta_search_country),
            "gamma_employment": float(gamma_employment_country),
            "gamma_temp": float(gamma_temp_country),
            "adstock_alpha_tv": float(adstock_alpha_tv),
            "adstock_alpha_search": float(adstock_alpha_search),
            "hill_lam_tv": float(hill_lam_tv),
            "hill_lam_search": float(hill_lam_search),
            "hill_s": float(hill_s),
            "fourier_coeffs": fourier_coeffs.tolist(),
        },
        "state_level": {s: {k: float(v) for k, v in p.items()} 
                       for s, p in state_params.items()}
    }
    
    import json
    with open(OUTPUT_DIR / "ground_truth_parameters.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
    
    return result_df


def main() -> None:
    """Main function to generate all data."""
    console.print("[bold green]Generating synthetic MMM-Chronos dataset[/bold green]")
    
    # Generate weekly dates (4 years)
    weeks = generate_weekly_dates(start_year=2020, num_years=4)
    console.print(f"[green]✓[/green] Generated {len(weeks)} weeks ({len(weeks)/52:.1f} years)")
    
    # Define file paths
    temp_file = OUTPUT_DIR / "temperature_data.csv"
    emp_file = OUTPUT_DIR / "employment_data.csv"
    media_file = OUTPUT_DIR / "media_spend.csv"
    sales_file = OUTPUT_DIR / "mmm_chronos_data.csv"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Fetch or load temperature data
        if temp_file.exists():
            task1 = progress.add_task("Loading existing temperature data...", total=None)
            temp_df = pl.read_csv(temp_file)
            progress.update(task1, completed=True)
            console.print(f"[blue]↻[/blue] Loaded temperature data from cache: {temp_df.shape}")
        else:
            task1 = progress.add_task("Fetching temperature data from Open-Meteo...", total=None)
            # Open-Meteo is free and doesn't require authentication
            temp_df = fetch_temperature_data_openmeteo(US_STATES, weeks)
            temp_df.write_csv(temp_file)
            progress.update(task1, completed=True)
            console.print(f"[green]✓[/green] Temperature data downloaded: {temp_df.shape}")
        
        # Fetch or load employment data
        if emp_file.exists():
            task2 = progress.add_task("Loading existing employment data...", total=None)
            emp_df = pl.read_csv(emp_file)
            progress.update(task2, completed=True)
            console.print(f"[blue]↻[/blue] Loaded employment data from cache: {emp_df.shape}")
        else:
            task2 = progress.add_task("Fetching employment data from FRED...", total=None)
            # Note: Can provide FRED API key via environment variable FRED_API_KEY if needed
            fred_api_key = os.getenv("FRED_API_KEY", None)
            emp_df = fetch_fred_unemployment_data(US_STATES, weeks, api_key=fred_api_key)
            emp_df.write_csv(emp_file)
            progress.update(task2, completed=True)
            console.print(f"[green]✓[/green] Employment data downloaded: {emp_df.shape}")
        
        # Generate media spend data
        task3 = progress.add_task("Generating media spend data...", total=None)
        media_df = generate_media_spend_data(US_STATES, weeks)
        media_df.write_csv(media_file)
        progress.update(task3, completed=True)
        console.print(f"[green]✓[/green] Media spend data: {media_df.shape}")
        
        # Generate sales data
        task4 = progress.add_task("Generating sales data...", total=None)
        sales_df = generate_sales_data(temp_df, emp_df, media_df)
        sales_df.write_csv(sales_file)
        progress.update(task4, completed=True)
        console.print(f"[green]✓[/green] Sales data: {sales_df.shape}")
    
    console.print(f"\n[bold green]✓ All data generated successfully![/bold green]")
    console.print(f"Output directory: {OUTPUT_DIR}")
    console.print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        size_mb = file.stat().st_size / (1024 * 1024)
        console.print(f"  • {file.name} ({size_mb:.2f} MB)")
    console.print(f"  • ground_truth_parameters.json")


if __name__ == "__main__":
    main()

