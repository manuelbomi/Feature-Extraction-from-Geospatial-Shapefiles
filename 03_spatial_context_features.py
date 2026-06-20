"""
03_spatial_context_features.py
================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Compute spatial CONTEXT features for each field — characteristics that
    depend on what's AROUND the field, not just the field itself.

    These features are essential for geospatial ML because:
    - A field's productivity is influenced by its neighbors' crop rotation
    - Disease/pest spread depends on neighbor crop types
    - Fragmentation index predicts mechanization difficulty
    - Cluster patterns reveal management zones

FEATURES COMPUTED:
    dist_to_nearest_m     — Distance to nearest neighboring field boundary
    n_neighbors_500m      — Count of fields with centroids within 500m
    neighbor_mean_area_ha — Mean area of neighbors within 500m
    neighbor_sum_area_ha  — Total area of neighbors within 500m
    dominant_neighbor_crop— Most common crop type among 500m neighbors
    edge_density          — perimeter / sqrt(area) — boundary complexity per size
    fragmentation_index   — Measures isolation: low = highly fragmented landscape
    local_crop_diversity  — Shannon entropy of crop types among 500m neighbors

SPATIAL INDEX:
    Uses Shapely's STRtree (Sort-Tile-Recursive tree) for O(n log n)
    nearest neighbor queries instead of O(n²) brute force.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import STRtree
from shapely.geometry import Point
import math
import os

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

data_path = "data/agricultural_fields.geojson"
gdf = gpd.read_file(data_path)
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

# Project to UTM for metric operations
gdf_utm = gdf.to_crs("EPSG:32611")

n = len(gdf_utm)
print(f"Computing spatial context features for {n} fields...")

# ---------------------------------------------------------------------------
# Build Spatial Index (STRtree)
# ---------------------------------------------------------------------------

# Build list of all field geometries for the tree
all_geoms = list(gdf_utm.geometry)
tree = STRtree(all_geoms)

print("Spatial index (STRtree) built successfully")

# ---------------------------------------------------------------------------
# Compute Spatial Context Features
# ---------------------------------------------------------------------------

NEIGHBOR_RADIUS_M = 500  # 500m radius for neighborhood

features = []

for i, row in gdf_utm.iterrows():
    field_id = gdf.iloc[i]["field_id"]
    own_geom = row.geometry
    own_centroid = own_geom.centroid
    own_area = own_geom.area / 10000  # ha
    own_perimeter = own_geom.length

    # --- Distance to nearest neighbor ---
    # Query the STRtree for candidates within a generous bounding box
    # then compute exact distance to each
    candidates = tree.query(own_geom.buffer(NEIGHBOR_RADIUS_M * 2))

    min_dist = float("inf")
    neighbor_areas = []
    neighbor_crops = []

    for j in candidates:
        if j == i:  # Skip self
            continue
        other_geom = all_geoms[j]
        other_centroid = other_geom.centroid
        other_crop = gdf.iloc[j]["crop_type"]
        other_area = other_geom.area / 10000

        # Distance from own centroid to other field's boundary
        dist_m = own_centroid.distance(other_geom)

        if dist_m < min_dist:
            min_dist = dist_m

        # Collect all neighbors within NEIGHBOR_RADIUS_M (centroid-to-centroid)
        centroid_dist = own_centroid.distance(other_centroid)
        if centroid_dist <= NEIGHBOR_RADIUS_M:
            neighbor_areas.append(other_area)
            neighbor_crops.append(other_crop)

    # Handle edge case: no neighbors found
    if min_dist == float("inf"):
        min_dist = 0.0

    # Neighbor statistics
    n_neighbors = len(neighbor_areas)
    neighbor_mean_area = np.mean(neighbor_areas) if neighbor_areas else 0.0
    neighbor_sum_area = np.sum(neighbor_areas) if neighbor_areas else 0.0

    # Dominant neighbor crop (most frequent)
    if neighbor_crops:
        from collections import Counter
        crop_counts = Counter(neighbor_crops)
        dominant_crop = crop_counts.most_common(1)[0][0]
    else:
        dominant_crop = "isolated"

    # Local crop diversity — Shannon entropy
    # H = -sum(p_i * log2(p_i)) where p_i is proportion of each crop type
    if neighbor_crops:
        from collections import Counter
        crop_counts = Counter(neighbor_crops)
        total = sum(crop_counts.values())
        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in crop_counts.values()
        )
    else:
        entropy = 0.0

    # Edge density: perimeter / sqrt(area) — normalized boundary length
    # Higher = more complex boundary relative to size
    edge_density = own_perimeter / math.sqrt(own_geom.area) if own_geom.area > 0 else 0.0

    # Fragmentation index: fields with no neighbors within radius are "isolated"
    fragmentation_index = 1.0 / (1.0 + n_neighbors)  # [0,1] — 1 = isolated

    features.append({
        "field_id":               field_id,
        "dist_to_nearest_m":      round(min_dist, 2),
        "n_neighbors_500m":        n_neighbors,
        "neighbor_mean_area_ha":   round(neighbor_mean_area, 4),
        "neighbor_sum_area_ha":    round(neighbor_sum_area, 4),
        "dominant_neighbor_crop":  dominant_crop,
        "edge_density":            round(edge_density, 6),
        "fragmentation_index":     round(fragmentation_index, 6),
        "local_crop_diversity_H":  round(entropy, 6),
    })

df_context = pd.DataFrame(features)

# ---------------------------------------------------------------------------
# Print and Export
# ---------------------------------------------------------------------------

print("\n=== Spatial Context Features ===")
print(df_context.to_string())

print("\n=== Summary Statistics ===")
print(df_context.drop(columns=["field_id", "dominant_neighbor_crop"]).describe().round(3).to_string())

print(f"\nIsolated fields (0 neighbors within {NEIGHBOR_RADIUS_M}m):")
isolated = df_context[df_context["n_neighbors_500m"] == 0]
if len(isolated) > 0:
    print(isolated[["field_id", "dist_to_nearest_m"]].to_string())
else:
    print("  None — all fields have at least one neighbor within 500m")

print("\nDominant neighbor crop distribution:")
print(df_context["dominant_neighbor_crop"].value_counts().to_string())

df_context.to_csv("spatial_context_features.csv", index=False)
print("\nExported: spatial_context_features.csv")

print("\n=== Script 03 Complete ===")
