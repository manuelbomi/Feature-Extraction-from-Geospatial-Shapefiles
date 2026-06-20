"""
05_time_series_feature_extraction.py
======================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Extract time series features from multi-year yield data.
    Each field has yield measurements for 2021-2025. These 5 data points
    per field can be transformed into informative features for ML:
    - Trend: Is this field's yield improving or declining?
    - Stability: Does it have consistent yield or high variability?
    - Extremes: What were the best and worst years?
    - Anomalies: Were there unusual years that deviate from the trend?

FEATURES EXTRACTED:
    mean_yield          — Average yield across all years
    std_yield           — Standard deviation (yield stability)
    cv_yield            — Coefficient of variation = std/mean (normalized variability)
    trend_slope         — Linear regression slope (units/year) — positive = improving
    trend_r2            — R² of linear fit — how strong is the trend?
    min_yield           — Worst year yield
    max_yield           — Best year yield
    yield_range         — max - min (yield spread)
    best_year           — Year of highest yield
    worst_year          — Year of lowest yield
    n_anomaly_years     — Years where yield deviated >2σ from mean
    anomaly_flag        — Boolean: any anomaly year detected
    last_year_trend     — Difference between last and first year (overall direction)
    recent_vs_historical— Last 2 years average vs first 3 years average

TIME SERIES ANALYSIS:
    With only 5 data points, classical time series methods (ARIMA, etc.)
    are inappropriate. Instead we extract hand-crafted features that capture
    the information content of the short time series.

    For longer time series (10+ years), consider:
    - tsfresh library for automated time series feature extraction
    - ROCKET/MiniRocket for convolutional time series features
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import os

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

data_path = "data/agricultural_fields.geojson"
gdf = gpd.read_file(data_path)
df = gdf.drop(columns=["geometry"])

# Identify yield columns
yield_cols = sorted([c for c in df.columns if c.startswith("yield_")])
years = [int(c.split("_")[1]) for c in yield_cols]

print(f"Time series analysis for {len(df)} fields")
print(f"Yield columns: {yield_cols}")
print(f"Years: {years}")

# ---------------------------------------------------------------------------
# Extract Time Series Features
# ---------------------------------------------------------------------------

features = []

for idx, row in df.iterrows():
    field_id = row["field_id"]
    crop = row["crop_type"]

    # Extract the yield time series as a numpy array
    yields = np.array([row[c] for c in yield_cols], dtype=float)

    # Handle missing values
    valid_mask = ~np.isnan(yields)
    if valid_mask.sum() < 2:
        features.append({
            "field_id": field_id,
            "crop_type": crop,
            "mean_yield": np.nan,
            "insufficient_data": True,
        })
        continue

    valid_yields = yields[valid_mask]
    valid_years = np.array(years)[valid_mask]

    # --- Basic statistics ---
    mean_yield = np.mean(valid_yields)
    std_yield = np.std(valid_yields, ddof=1)  # Sample std
    cv_yield = (std_yield / mean_yield) if mean_yield != 0 else 0.0  # % variability
    min_yield = np.min(valid_yields)
    max_yield = np.max(valid_yields)
    yield_range = max_yield - min_yield

    # --- Best and worst year ---
    best_year_idx = np.argmax(valid_yields)
    worst_year_idx = np.argmin(valid_yields)
    best_year = valid_years[best_year_idx]
    worst_year = valid_years[worst_year_idx]

    # --- Linear trend (slope) ---
    # Fit y = a*year + b using least squares
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        valid_years, valid_yields
    )
    trend_slope = slope  # Units per year (e.g., tonnes/ha/year)
    trend_r2 = r_value ** 2
    trend_p_value = p_value
    trend_significant = p_value < 0.05  # Statistically significant trend

    # --- Anomaly detection ---
    # An anomaly year is one where yield deviates more than 2σ from the mean
    z_scores = np.abs((valid_yields - mean_yield) / std_yield) if std_yield > 0 else np.zeros_like(valid_yields)
    anomaly_mask = z_scores > 2.0
    n_anomaly_years = int(anomaly_mask.sum())
    anomaly_flag = n_anomaly_years > 0
    anomaly_years = [int(y) for y in valid_years[anomaly_mask]]

    # --- Recent vs historical comparison ---
    if len(valid_yields) >= 3:
        historical_yields = valid_yields[:-2]  # First n-2 years
        recent_yields = valid_yields[-2:]        # Last 2 years
        recent_vs_historical = np.mean(recent_yields) - np.mean(historical_yields)
    else:
        recent_vs_historical = 0.0

    # --- Last year trend (simple direction) ---
    last_year_trend = valid_yields[-1] - valid_yields[0]  # End - start

    # --- Year-over-year changes ---
    yoy_changes = np.diff(valid_yields)
    mean_yoy_change = np.mean(yoy_changes) if len(yoy_changes) > 0 else 0.0
    max_yoy_increase = np.max(yoy_changes) if len(yoy_changes) > 0 else 0.0
    max_yoy_decrease = np.min(yoy_changes) if len(yoy_changes) > 0 else 0.0

    features.append({
        "field_id":               field_id,
        "crop_type":              crop,
        "mean_yield":             round(float(mean_yield), 4),
        "std_yield":              round(float(std_yield), 4),
        "cv_yield":               round(float(cv_yield), 4),
        "trend_slope":            round(float(trend_slope), 6),
        "trend_r2":               round(float(trend_r2), 6),
        "trend_p_value":          round(float(trend_p_value), 6),
        "trend_significant":      int(trend_significant),
        "min_yield":              round(float(min_yield), 4),
        "max_yield":              round(float(max_yield), 4),
        "yield_range":            round(float(yield_range), 4),
        "best_year":              int(best_year),
        "worst_year":             int(worst_year),
        "n_anomaly_years":        n_anomaly_years,
        "anomaly_flag":           int(anomaly_flag),
        "anomaly_years":          str(anomaly_years),
        "last_year_trend":        round(float(last_year_trend), 4),
        "recent_vs_historical":   round(float(recent_vs_historical), 4),
        "mean_yoy_change":        round(float(mean_yoy_change), 4),
        "max_yoy_increase":       round(float(max_yoy_increase), 4),
        "max_yoy_decrease":       round(float(max_yoy_decrease), 4),
        "insufficient_data":      False,
    })

df_ts = pd.DataFrame(features)

# ---------------------------------------------------------------------------
# Print Results
# ---------------------------------------------------------------------------

print("\n=== Time Series Feature Matrix ===")
display_cols = ["field_id", "crop_type", "mean_yield", "std_yield", "cv_yield",
                "trend_slope", "trend_r2", "best_year", "worst_year",
                "anomaly_flag", "n_anomaly_years"]
print(df_ts[display_cols].to_string())

print("\n=== Trend Analysis ===")
trending_up = df_ts[df_ts["trend_slope"] > 0]
trending_down = df_ts[df_ts["trend_slope"] < 0]
significant = df_ts[df_ts["trend_significant"] == 1]

print(f"Fields with improving yield trend (+slope): {len(trending_up)}")
print(f"Fields with declining yield trend (-slope): {len(trending_down)}")
print(f"Fields with statistically significant trend (p<0.05): {len(significant)}")

print("\n=== Anomaly Detection ===")
anomalous = df_ts[df_ts["anomaly_flag"] == 1]
if len(anomalous) > 0:
    print(f"Fields with yield anomalies (>2σ): {len(anomalous)}")
    print(anomalous[["field_id", "crop_type", "mean_yield", "std_yield",
                      "n_anomaly_years", "anomaly_years"]].to_string())
else:
    print("No yield anomalies detected in any field")

print("\n=== Top 5 Fields by Yield Improvement Trend ===")
top_trend = df_ts.nlargest(5, "trend_slope")[
    ["field_id", "crop_type", "mean_yield", "trend_slope", "trend_r2"]
]
print(top_trend.to_string(index=False))

print("\n=== Most Stable Yields (lowest CV) ===")
most_stable = df_ts.nsmallest(5, "cv_yield")[
    ["field_id", "crop_type", "mean_yield", "cv_yield", "std_yield"]
]
print(most_stable.to_string(index=False))

# Export
df_ts.to_csv("time_series_features.csv", index=False)
print("\nExported: time_series_features.csv")

print("\n=== Script 05 Complete ===")
