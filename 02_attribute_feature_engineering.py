"""
02_attribute_feature_engineering.py
=====================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Engineer ML-ready features from shapefile attribute columns.
    Demonstrates a complete sklearn Pipeline for geospatial data attributes,
    including categorical encoding, binning, ratio features, and normalization.

FEATURE ENGINEERING TECHNIQUES:
    1. Label Encoding — crop_type → integer category code
    2. One-Hot Encoding — crop_type → binary dummy columns
    3. Ordinal Binning — area_ha → size class (small/medium/large/xlarge)
    4. Ratio Features — yield per area, slope per area
    5. Lag Features — year-over-year yield deltas for time trends
    6. Min-Max Normalization — scale all numerics to [0, 1]
    7. sklearn ColumnTransformer — combine all transformations
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, MinMaxScaler, OrdinalEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

data_path = "data/agricultural_fields.geojson"
gdf = gpd.read_file(data_path)
# Drop geometry for attribute-only analysis
df = gdf.drop(columns=["geometry"])

print(f"Loaded {len(df)} fields with {len(df.columns)} attribute columns")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes.to_string()}")
print(f"\nCrop type distribution:\n{df['crop_type'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# FEATURE 1: Label Encoding for crop_type
# ---------------------------------------------------------------------------

le = LabelEncoder()
df["crop_type_encoded"] = le.fit_transform(df["crop_type"].str.lower().fillna("unknown"))

crop_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\nCrop type label encoding:\n{crop_mapping}")

# ---------------------------------------------------------------------------
# FEATURE 2: One-Hot Encoding for crop_type and soil_type
# ---------------------------------------------------------------------------

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
crop_dummies = ohe.fit_transform(df[["crop_type"]])
crop_feature_names = [f"crop__{c}" for c in ohe.categories_[0]]
df_crop_ohe = pd.DataFrame(crop_dummies, columns=crop_feature_names, index=df.index)

ohe_soil = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
soil_dummies = ohe_soil.fit_transform(df[["soil_type"]])
soil_feature_names = [f"soil__{c}" for c in ohe_soil.categories_[0]]
df_soil_ohe = pd.DataFrame(soil_dummies, columns=soil_feature_names, index=df.index)

print(f"\nOne-hot crop features: {crop_feature_names}")
print(f"One-hot soil features: {soil_feature_names}")

# ---------------------------------------------------------------------------
# FEATURE 3: Ordinal Binning — area size classes
# ---------------------------------------------------------------------------

size_bins = [0, 20, 40, 60, float("inf")]
size_labels = ["small", "medium", "large", "xlarge"]
df["area_size_class"] = pd.cut(df["area_ha"], bins=size_bins, labels=size_labels)
df["area_size_code"] = df["area_size_class"].cat.codes  # 0,1,2,3

print(f"\nArea size class distribution:\n{df['area_size_class'].value_counts().to_string()}")

# ---------------------------------------------------------------------------
# FEATURE 4: Ratio Features
# ---------------------------------------------------------------------------

yield_cols = [c for c in df.columns if c.startswith("yield_")]

# Mean yield
df["mean_yield"] = df[yield_cols].mean(axis=1)

# Yield per hectare (already in per-hectare units for crops, normalize by area)
# For crops like corn measured in tonnes: yield_efficiency = yield / area
df["yield_efficiency"] = df["mean_yield"] / df["area_ha"]

# Slope relative to area (high-slope + large-area = challenging field)
df["slope_area_ratio"] = df["slope_pct"] / (df["area_ha"] + 1)  # +1 to avoid div by 0

# Irrigation type efficiency score (domain knowledge encoding)
irrigation_scores = {
    "drip": 1.0,        # Most efficient
    "sprinkler": 0.8,
    "center_pivot": 0.75,
    "furrow": 0.5,
    "flood": 0.3,       # Least efficient
}
df["irrigation_efficiency_score"] = df["irrigation_type"].map(irrigation_scores).fillna(0.5)

print("\nRatio features summary:")
print(df[["field_id", "mean_yield", "yield_efficiency",
          "slope_area_ratio", "irrigation_efficiency_score"]].to_string())

# ---------------------------------------------------------------------------
# FEATURE 5: Year-over-Year Change Features (Lag Features)
# ---------------------------------------------------------------------------

# Sort yield columns chronologically
yield_cols_sorted = sorted(yield_cols)  # ['yield_2021', ..., 'yield_2025']

for i in range(1, len(yield_cols_sorted)):
    prev_year = yield_cols_sorted[i - 1]
    curr_year = yield_cols_sorted[i]
    col_name = f"yoy_change_{curr_year[-4:]}"
    df[col_name] = df[curr_year] - df[prev_year]

yoy_cols = [c for c in df.columns if c.startswith("yoy_change_")]
print(f"\nYear-over-year change features: {yoy_cols}")
print(df[["field_id"] + yoy_cols].to_string())

# ---------------------------------------------------------------------------
# FEATURE 6: sklearn ColumnTransformer Pipeline
# ---------------------------------------------------------------------------

# Select numeric columns for scaling
numeric_cols = ["area_ha", "slope_pct", "mean_yield", "yield_efficiency",
                "slope_area_ratio", "irrigation_efficiency_score",
                "area_size_code", "crop_type_encoded"] + yoy_cols

# Make sure all numeric cols exist
numeric_cols = [c for c in numeric_cols if c in df.columns]

# Build sklearn pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(), numeric_cols),
        ("crop_ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         ["crop_type"]),
        ("soil_ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
         ["soil_type"]),
    ],
    remainder="drop"  # Drop columns not specified above
)

# Fit and transform
df_filled = df[numeric_cols + ["crop_type", "soil_type"]].fillna(0)
feature_matrix = preprocessor.fit_transform(df_filled)

# Build feature names
crop_categories = preprocessor.named_transformers_["crop_ohe"].categories_[0]
soil_categories = preprocessor.named_transformers_["soil_ohe"].categories_[0]
all_feature_names = (
    numeric_cols
    + [f"crop__{c}" for c in crop_categories]
    + [f"soil__{c}" for c in soil_categories]
)

df_features = pd.DataFrame(
    feature_matrix,
    columns=all_feature_names,
    index=df.index
)
df_features.insert(0, "field_id", df["field_id"].values)

print(f"\n=== Attribute Feature Matrix (sklearn pipeline output) ===")
print(f"Shape: {df_features.shape}")
print(df_features.head(5).to_string())

# Export
df_features.to_csv("attribute_features.csv", index=False)
print("\nExported: attribute_features.csv")

print("\n=== Script 02 Complete ===")
