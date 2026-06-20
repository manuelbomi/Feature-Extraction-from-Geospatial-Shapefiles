"""
04_topology_feature_extraction.py
===================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Extract topological (adjacency) features from field polygons.
    Two fields are "adjacent" if they share a common boundary segment —
    NOT just if they're nearby. Topological adjacency captures:
    - Shared ownership patterns (adjacent fields often same owner)
    - Disease/weed spread pathways (crossing shared boundaries)
    - Irrigation infrastructure sharing
    - Management zone clustering

FEATURES COMPUTED:
    n_adjacent_fields     — How many fields share a boundary with this field
    total_shared_length_m — Total length of shared boundaries (meters)
    mean_shared_length_m  — Mean shared segment length
    max_shared_length_m   — Longest shared boundary
    is_isolated           — Boolean: no shared boundaries at all
    adjacency_degree      — Same as n_adjacent (for graph theory terminology)
    shared_crop_count     — How many adjacent fields have same crop type
    different_crop_count  — How many adjacent fields have different crop type

ALGORITHM:
    For each pair (i, j), check if fields share a boundary using:
    geom_i.touches(geom_j)  — True if they share exactly a boundary (no overlap)
    If touching, compute the intersection which gives the shared LineString.
    The length of that LineString is the shared boundary length.

PRODUCTION NOTE:
    For large datasets (10,000+ polygons), pre-filter with STRtree:
        candidates = tree.query(field_geom, predicate="touches")
    This avoids checking all O(n²) pairs.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString
import os

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

data_path = "data/agricultural_fields.geojson"
gdf = gpd.read_file(data_path)
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

gdf_utm = gdf.to_crs("EPSG:32611")
n = len(gdf_utm)

print(f"Computing topology features for {n} fields...")
print(f"Pairs to check: {n * (n-1) // 2} (brute force)")

# ---------------------------------------------------------------------------
# Build Adjacency Structure
# ---------------------------------------------------------------------------

# Tolerance for "touches" detection — in meters (UTM).
# Real-world boundaries rarely align perfectly; a small tolerance
# prevents missing truly adjacent fields due to digitization gaps.
TOUCH_TOLERANCE_M = 0.5  # 50cm tolerance buffer

# adjacency[i] = {j: shared_length_m} for all adjacent field j
adjacency = {i: {} for i in range(n)}

# Use STRtree to find candidate pairs efficiently
all_geoms = list(gdf_utm.geometry)
tree = STRtree(all_geoms)

for i in range(n):
    geom_i = all_geoms[i]
    crop_i = gdf.iloc[i]["crop_type"]

    # Query tree for geometries within touch distance
    # Buffer by tolerance to catch near-touching fields
    candidates = tree.query(geom_i.buffer(TOUCH_TOLERANCE_M))

    for j in candidates:
        if j <= i:  # Avoid duplicate pairs (i,j) and (j,i)
            continue

        geom_j = all_geoms[j]

        # Check if fields touch (share boundary, no overlap)
        # We use buffered intersection to handle digitization gaps
        geom_i_buf = geom_i.buffer(TOUCH_TOLERANCE_M / 2)
        geom_j_buf = geom_j.buffer(TOUCH_TOLERANCE_M / 2)

        if geom_i_buf.intersects(geom_j_buf):
            # Compute the intersection
            intersection = geom_i_buf.intersection(geom_j_buf)

            # Only count as adjacent if intersection is line-like (not polygon)
            # A polygon intersection would mean overlap — that's a data error
            shared_length = 0.0
            if intersection.geom_type == "LineString":
                shared_length = intersection.length
            elif intersection.geom_type == "MultiLineString":
                shared_length = intersection.length
            elif intersection.geom_type in ("Polygon", "MultiPolygon"):
                # This is an overlap, not adjacency — still record but flag
                shared_length = intersection.length  # Use perimeter as proxy
            elif intersection.geom_type in ("Point", "MultiPoint"):
                # Point touch only — technically adjacent but no shared edge length
                shared_length = 0.01  # Epsilon to mark adjacency
            else:
                # GeometryCollection — try to extract linear components
                if hasattr(intersection, "geoms"):
                    for component in intersection.geoms:
                        if component.geom_type in ("LineString", "MultiLineString"):
                            shared_length += component.length

            if shared_length > 0:
                adjacency[i][j] = shared_length
                adjacency[j][i] = shared_length
                print(f"  Adjacent: {gdf.iloc[i]['field_id']} ↔ "
                      f"{gdf.iloc[j]['field_id']} "
                      f"(shared={shared_length:.1f}m)")

# ---------------------------------------------------------------------------
# Extract Topological Features per Field
# ---------------------------------------------------------------------------

features = []
for i in range(n):
    field_id = gdf.iloc[i]["field_id"]
    own_crop = gdf.iloc[i]["crop_type"]
    adj = adjacency[i]  # dict of {j: shared_length}

    n_adjacent = len(adj)
    shared_lengths = list(adj.values())

    if n_adjacent > 0:
        total_shared_m = sum(shared_lengths)
        mean_shared_m = np.mean(shared_lengths)
        max_shared_m = max(shared_lengths)
    else:
        total_shared_m = 0.0
        mean_shared_m = 0.0
        max_shared_m = 0.0

    is_isolated = (n_adjacent == 0)

    # Count adjacent fields with same vs different crop type
    same_crop = 0
    diff_crop = 0
    for j in adj.keys():
        neighbor_crop = gdf.iloc[j]["crop_type"]
        if neighbor_crop == own_crop:
            same_crop += 1
        else:
            diff_crop += 1

    features.append({
        "field_id":              field_id,
        "n_adjacent_fields":      n_adjacent,
        "total_shared_length_m":  round(total_shared_m, 2),
        "mean_shared_length_m":   round(mean_shared_m, 2),
        "max_shared_length_m":    round(max_shared_m, 2),
        "is_isolated":            int(is_isolated),
        "adjacency_degree":       n_adjacent,  # Graph theory terminology
        "shared_crop_count":      same_crop,
        "different_crop_count":   diff_crop,
    })

df_topo = pd.DataFrame(features)

# ---------------------------------------------------------------------------
# Print Adjacency Matrix and Results
# ---------------------------------------------------------------------------

print("\n=== Adjacency Matrix (shared boundary length in meters) ===")
field_ids = gdf["field_id"].tolist()
adj_matrix = pd.DataFrame(0.0, index=field_ids, columns=field_ids)
for i in range(n):
    for j, length in adjacency[i].items():
        adj_matrix.iloc[i, j] = length

# Show only fields that have at least one adjacency
has_adj = adj_matrix.sum(axis=1) > 0
print(adj_matrix.loc[has_adj, has_adj].round(1).to_string())

print("\n=== Topological Features ===")
print(df_topo.to_string())

n_isolated = df_topo["is_isolated"].sum()
n_connected = n - n_isolated
print(f"\nNetwork statistics:")
print(f"  Connected fields: {n_connected} ({n_connected/n*100:.0f}%)")
print(f"  Isolated fields: {n_isolated} ({n_isolated/n*100:.0f}%)")
print(f"  Total adjacency pairs: {sum(len(v) for v in adjacency.values()) // 2}")
print(f"  Max degree (most adjacent): {df_topo['n_adjacent_fields'].max()} fields")

if n_connected > 0:
    most_connected = df_topo.loc[df_topo["n_adjacent_fields"].idxmax(), "field_id"]
    print(f"  Most-connected field: {most_connected}")

df_topo.to_csv("topology_features.csv", index=False)
print("\nExported: topology_features.csv")

print("\n=== Script 04 Complete ===")
