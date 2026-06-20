"""
06_ml_ready_feature_pipeline.py
=================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    End-to-end pipeline: load shapefile → extract all feature types →
    merge into single feature matrix → preprocess → train RandomForest →
    evaluate and report feature importances.

    This script is the culmination of scripts 01-05, demonstrating how all
    feature types work together for a classification task:
    PREDICT CROP TYPE FROM FIELD CHARACTERISTICS.

    Note: With only 20 samples, this is purely a demonstration of the
    pipeline architecture. In production, you'd have thousands of fields.

PIPELINE STAGES:
    1. Load GeoDataFrame
    2. Geometry features (from script 01 logic)
    3. Attribute features (from script 02 logic)
    4. Time series features (from script 05 logic)
    5. Merge all feature types into one matrix
    6. Handle missing values (median imputation)
    7. Feature scaling (StandardScaler)
    8. Train/test split (stratified)
    9. RandomForest classifier
    10. Evaluation: accuracy, confusion matrix, feature importances
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import os
from scipy import stats as scipy_stats

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripting
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# STEP 1: Load Data
# ---------------------------------------------------------------------------

data_path = "data/agricultural_fields.geojson"
gdf = gpd.read_file(data_path)
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

gdf_utm = gdf.to_crs("EPSG:32611")
print(f"Loaded {len(gdf)} fields")

# ---------------------------------------------------------------------------
# STEP 2: Geometry Features
# ---------------------------------------------------------------------------

def extract_geometry_features(gdf_utm):
    rows = []
    for i, row in gdf_utm.iterrows():
        geom = row.geometry
        field_id = gdf.iloc[i]["field_id"]
        area_m2 = geom.area
        perimeter_m = geom.length
        convex_hull = geom.convex_hull

        compactness = (4 * math.pi * area_m2 / perimeter_m**2) if perimeter_m > 0 else 0
        minx, miny, maxx, maxy = geom.bounds
        bbox_w = maxx - minx
        bbox_h = maxy - miny
        elongation = (max(bbox_w, bbox_h) / min(bbox_w, bbox_h)) if min(bbox_w, bbox_h) > 0 else 1
        solidity = (area_m2 / convex_hull.area) if convex_hull.area > 0 else 0
        convexity = (convex_hull.length / perimeter_m) if perimeter_m > 0 else 0
        n_verts = len(list(geom.exterior.coords)) - 1
        bbox_fill = (area_m2 / (bbox_w * bbox_h)) if (bbox_w * bbox_h) > 0 else 0
        edge_density = perimeter_m / math.sqrt(area_m2) if area_m2 > 0 else 0

        rows.append({
            "field_id":        field_id,
            "geom_area_ha":    area_m2 / 10000,
            "geom_perim_km":   perimeter_m / 1000,
            "geom_compactness":compactness,
            "geom_elongation": elongation,
            "geom_solidity":   solidity,
            "geom_convexity":  convexity,
            "geom_n_vertices": n_verts,
            "geom_bbox_fill":  bbox_fill,
            "geom_edge_density": edge_density,
        })
    return pd.DataFrame(rows)

df_geom = extract_geometry_features(gdf_utm)
print(f"Geometry features: {df_geom.shape[1] - 1} features")

# ---------------------------------------------------------------------------
# STEP 3: Attribute Features
# ---------------------------------------------------------------------------

df_attr = gdf.drop(columns=["geometry"]).copy()

# Encode irrigation_type as score
irrigation_score = {
    "drip": 1.0, "sprinkler": 0.8, "center_pivot": 0.75,
    "furrow": 0.5, "flood": 0.3
}
df_attr["irr_score"] = df_attr["irrigation_type"].map(irrigation_score).fillna(0.5)

yield_cols = sorted([c for c in df_attr.columns if c.startswith("yield_")])
df_attr["attr_mean_yield"] = df_attr[yield_cols].mean(axis=1)
df_attr["attr_slope_pct"] = df_attr["slope_pct"]
df_attr["attr_area_ha"] = df_attr["area_ha"]
df_attr["attr_irr_score"] = df_attr["irr_score"]

# Soil type ordinal encoding (sand=0, loam=0.5, clay=1.0 for drainage)
soil_drainage = {
    "sandy": 0.9, "loamy_sand": 0.75, "sandy_loam": 0.65,
    "loam": 0.5, "silty_loam": 0.4, "silty_clay": 0.25,
    "clay_loam": 0.2, "clay": 0.1
}
df_attr["attr_soil_drainage"] = df_attr["soil_type"].map(soil_drainage).fillna(0.5)

attr_feature_cols = ["field_id", "attr_mean_yield", "attr_slope_pct",
                     "attr_area_ha", "attr_irr_score", "attr_soil_drainage"]
df_attr_features = df_attr[attr_feature_cols]
print(f"Attribute features: {df_attr_features.shape[1] - 1} features")

# ---------------------------------------------------------------------------
# STEP 4: Time Series Features
# ---------------------------------------------------------------------------

def extract_ts_features(df_attr, yield_cols):
    years = [int(c.split("_")[1]) for c in yield_cols]
    rows = []
    for idx, row in df_attr.iterrows():
        ys = np.array([row[c] for c in yield_cols], dtype=float)
        valid = ~np.isnan(ys)
        if valid.sum() >= 2:
            yv = ys[valid]
            yr = np.array(years)[valid]
            slope, _, r, p, _ = scipy_stats.linregress(yr, yv)
            cv = yv.std(ddof=1) / yv.mean() if yv.mean() != 0 else 0
            z = np.abs((yv - yv.mean()) / yv.std(ddof=1)) if yv.std(ddof=1) > 0 else np.zeros_like(yv)
            rows.append({
                "field_id": row["field_id"],
                "ts_mean_yield": yv.mean(),
                "ts_cv_yield": cv,
                "ts_trend_slope": slope,
                "ts_trend_r2": r**2,
                "ts_min_yield": yv.min(),
                "ts_max_yield": yv.max(),
                "ts_yield_range": yv.max() - yv.min(),
                "ts_n_anomalies": int((z > 2.0).sum()),
                "ts_last_vs_first": float(yv[-1] - yv[0]),
            })
        else:
            rows.append({"field_id": row["field_id"]})
    return pd.DataFrame(rows)

df_ts = extract_ts_features(df_attr, yield_cols)
print(f"Time series features: {df_ts.shape[1] - 1} features")

# ---------------------------------------------------------------------------
# STEP 5: Merge All Features
# ---------------------------------------------------------------------------

df_merged = df_geom.merge(df_attr_features, on="field_id", how="left")
df_merged = df_merged.merge(df_ts, on="field_id", how="left")

# Add target label
label_df = gdf[["field_id", "crop_type"]].copy()
df_merged = df_merged.merge(label_df, on="field_id", how="left")

print(f"\nMerged feature matrix: {df_merged.shape}")
print(f"Features: {df_merged.shape[1] - 2}")  # -2 for field_id and crop_type

# ---------------------------------------------------------------------------
# STEP 6: Prepare Feature Matrix and Labels
# ---------------------------------------------------------------------------

feature_cols = [c for c in df_merged.columns
                if c not in ("field_id", "crop_type")]

X = df_merged[feature_cols].copy()
y_raw = df_merged["crop_type"].str.lower()

# Label encode target
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

print(f"\nTarget classes: {list(class_names)}")
print(f"Class distribution: {pd.Series(y_raw).value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# STEP 7: Preprocessing Pipeline
# ---------------------------------------------------------------------------

preproc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill NaN with column median
    ("scaler", StandardScaler()),                   # Standardize to zero mean, unit variance
])

X_processed = preproc.fit_transform(X)
print(f"\nProcessed feature matrix: {X_processed.shape}")

# ---------------------------------------------------------------------------
# STEP 8: Train RandomForest with Cross-Validation
# ---------------------------------------------------------------------------

# With only 20 samples, use stratified k-fold cross-validation
# instead of a simple train/test split
n_splits = min(5, y.min() if hasattr(y, "min") else 3)
n_splits = max(2, n_splits)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=1,
    class_weight="balanced",  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
)

# Reduce splits if not enough samples per class
unique, counts = np.unique(y, return_counts=True)
min_class_count = counts.min()
n_splits = min(n_splits, min_class_count)

if n_splits >= 2:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_processed, y, cv=cv, scoring="accuracy")
    print(f"\nCross-validation ({n_splits}-fold) accuracy: "
          f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"CV scores per fold: {cv_scores.round(3)}")
else:
    print(f"\nInsufficient samples for cross-validation (min class size={min_class_count})")
    cv_scores = np.array([0.0])

# Fit on full dataset for feature importances
rf.fit(X_processed, y)
y_pred = rf.predict(X_processed)
train_accuracy = accuracy_score(y, y_pred)
print(f"Training accuracy (in-sample): {train_accuracy:.3f}")

# ---------------------------------------------------------------------------
# STEP 9: Feature Importances
# ---------------------------------------------------------------------------

importances = pd.Series(rf.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=False)

print("\n=== TOP 15 FEATURE IMPORTANCES ===")
for feat, imp in importances.head(15).items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<35} {imp:.4f}  {bar}")

# ---------------------------------------------------------------------------
# STEP 10: Classification Report
# ---------------------------------------------------------------------------

print("\n=== CLASSIFICATION REPORT (training set) ===")
print(classification_report(
    y, y_pred,
    target_names=class_names,
    zero_division=0
))

# ---------------------------------------------------------------------------
# STEP 11: Feature Importance Plot
# ---------------------------------------------------------------------------

top_n = min(15, len(importances))
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
importances.head(top_n).plot(kind="barh", ax=ax, color=colors[::-1])
ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=11)
ax.set_title(
    "RandomForest Feature Importances — Crop Type Classification\n"
    f"20 Agricultural Fields | {len(feature_cols)} Features | Emmanuel Oyekanlu",
    fontsize=12, fontweight="bold"
)
ax.grid(axis="x", alpha=0.4)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150, bbox_inches="tight")
print("\nSaved: feature_importances.png")

# ---------------------------------------------------------------------------
# STEP 12: Export final feature matrix
# ---------------------------------------------------------------------------

df_final = df_merged[["field_id", "crop_type"] + feature_cols].copy()
df_final.to_csv("full_feature_matrix.csv", index=False)
print("Saved: full_feature_matrix.csv")

print(f"\n=== ML Pipeline Complete ===")
print(f"Feature matrix: {df_final.shape[0]} samples × {len(feature_cols)} features")
print(f"Cross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Top feature: {importances.index[0]} ({importances.iloc[0]:.4f})")
print("\n=== Script 06 Complete ===")
