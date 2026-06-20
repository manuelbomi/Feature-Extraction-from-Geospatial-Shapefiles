"""
01_geometry_feature_extraction.py
==================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

PURPOSE:
    Extract shape-based numerical features from polygon geometries.
    These features capture the form, size, and complexity of each field —
    information that is predictive for many agricultural ML tasks:
    - Compact, rectangular fields are easier to mechanize
    - Highly elongated fields suggest terrain-following planting
    - Fields with low solidity (jagged edges) often have drainage channels
    - Very large n_vertices suggests complex ownership boundaries

FEATURES COMPUTED:
    area_ha              — Field area in hectares (metric)
    perimeter_km         — Perimeter in kilometers
    compactness          — 4π·A/P² = 1.0 for a perfect circle, <1 for all real shapes
    elongation           — Ratio of longest to shortest bounding box dimension
    convex_hull_area_ha  — Area of the convex hull
    solidity             — area / convex_hull_area. Low = concave/irregular
    convexity            — convex_hull_perimeter / perimeter. Low = jagged edges
    n_vertices           — Number of polygon vertices (shape complexity)
    bbox_area_ha         — Bounding box area
    bbox_fill_ratio      — area / bbox_area (how well the field fills its bbox)
    centroid_lon         — Centroid longitude (WGS84)
    centroid_lat         — Centroid latitude (WGS84)
    bbox_width_m         — Bounding box east-west extent in meters
    bbox_height_m        — Bounding box north-south extent in meters

NOTE ON CRS:
    All metric calculations use a projected CRS (EPSG:32611 — UTM Zone 11N
    for Imperial Valley, CA). Never compute area/perimeter in degrees!
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import math
import os

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------

data_path = "data/agricultural_fields.geojson"
gdf = gpd.read_file(data_path)

if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")

print(f"Loaded {len(gdf)} fields | CRS: {gdf.crs}")
print(f"Columns: {list(gdf.columns)}")

# Project to UTM for metric calculations
gdf_utm = gdf.to_crs("EPSG:32611")

# ---------------------------------------------------------------------------
# Feature Extraction Functions
# ---------------------------------------------------------------------------

def compute_compactness(area_m2: float, perimeter_m: float) -> float:
    """
    Polsby-Popper compactness score.
    Range: (0, 1] — circle scores 1.0, elongated/irregular shapes < 1.0.
    Used in electoral district analysis and agricultural field studies.
    """
    if perimeter_m <= 0:
        return 0.0
    return (4 * math.pi * area_m2) / (perimeter_m ** 2)

def compute_elongation(geom) -> float:
    """
    Ratio of bounding box width to height (max/min).
    1.0 = square bounding box, >1 = elongated east-west or north-south.
    """
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny
    if min(width, height) == 0:
        return float("inf")
    return max(width, height) / min(width, height)

def count_vertices(geom) -> int:
    """Count exterior ring vertices (excluding closing vertex)."""
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        # Shapely closes the ring (first == last), so subtract 1
        return len(coords) - 1
    elif geom.geom_type == "MultiPolygon":
        return sum(len(list(p.exterior.coords)) - 1 for p in geom.geoms)
    return 0

# ---------------------------------------------------------------------------
# Extract All Geometry Features
# ---------------------------------------------------------------------------

features = []

for idx, row in gdf_utm.iterrows():
    geom = row.geometry
    field_id = gdf.iloc[idx]["field_id"]

    # Basic measurements
    area_m2 = geom.area
    area_ha = area_m2 / 10000
    perimeter_m = geom.length
    perimeter_km = perimeter_m / 1000

    # Compactness
    compactness = compute_compactness(area_m2, perimeter_m)

    # Elongation from bounding box
    elongation = compute_elongation(geom)

    # Bounding box dimensions
    minx, miny, maxx, maxy = geom.bounds
    bbox_width_m = maxx - minx
    bbox_height_m = maxy - miny
    bbox_area_m2 = bbox_width_m * bbox_height_m
    bbox_area_ha = bbox_area_m2 / 10000
    bbox_fill_ratio = (area_m2 / bbox_area_m2) if bbox_area_m2 > 0 else 0.0

    # Convex hull features
    convex_hull = geom.convex_hull
    convex_hull_area_m2 = convex_hull.area
    convex_hull_area_ha = convex_hull_area_m2 / 10000
    convex_hull_perimeter_m = convex_hull.length

    # Solidity: area / convex_hull_area
    # High solidity (→1) = mostly convex (rectangular fields)
    # Low solidity (<0.8) = concave, irregular shapes
    solidity = (area_m2 / convex_hull_area_m2) if convex_hull_area_m2 > 0 else 0.0

    # Convexity: convex_hull_perimeter / actual_perimeter
    # High convexity (→1) = smooth boundary
    # Low convexity = jagged, indented boundary
    convexity = (convex_hull_perimeter_m / perimeter_m) if perimeter_m > 0 else 0.0

    # Vertex count (shape complexity)
    n_vertices = count_vertices(geom)

    # Centroid in WGS84
    centroid_utm = geom.centroid
    centroid_wgs84 = gpd.GeoSeries(
        [centroid_utm], crs="EPSG:32611"
    ).to_crs("EPSG:4326").iloc[0]
    centroid_lon = centroid_wgs84.x
    centroid_lat = centroid_wgs84.y

    features.append({
        "field_id":           field_id,
        "area_ha":            round(area_ha, 4),
        "perimeter_km":       round(perimeter_km, 4),
        "compactness":        round(compactness, 6),
        "elongation":         round(elongation, 4),
        "convex_hull_area_ha":round(convex_hull_area_ha, 4),
        "solidity":           round(solidity, 6),
        "convexity":          round(convexity, 6),
        "n_vertices":         n_vertices,
        "bbox_area_ha":       round(bbox_area_ha, 4),
        "bbox_fill_ratio":    round(bbox_fill_ratio, 6),
        "bbox_width_m":       round(bbox_width_m, 2),
        "bbox_height_m":      round(bbox_height_m, 2),
        "centroid_lon":       round(centroid_lon, 6),
        "centroid_lat":       round(centroid_lat, 6),
    })

df_features = pd.DataFrame(features)

# ---------------------------------------------------------------------------
# Print and Export
# ---------------------------------------------------------------------------

print("\n=== Geometry Feature Matrix ===")
print(df_features.to_string())

print("\n=== Feature Statistics ===")
print(df_features.drop(columns=["field_id"]).describe().round(4).to_string())

# Identify interesting fields
print("\n=== Notable Fields ===")
print(f"Most compact field: {df_features.loc[df_features['compactness'].idxmax(), 'field_id']} "
      f"(compactness={df_features['compactness'].max():.4f})")
print(f"Most elongated:     {df_features.loc[df_features['elongation'].idxmax(), 'field_id']} "
      f"(elongation={df_features['elongation'].max():.2f})")
print(f"Largest field:      {df_features.loc[df_features['area_ha'].idxmax(), 'field_id']} "
      f"({df_features['area_ha'].max():.1f} ha)")
print(f"Most complex shape: {df_features.loc[df_features['n_vertices'].idxmax(), 'field_id']} "
      f"({df_features['n_vertices'].max()} vertices)")
print(f"Lowest solidity:    {df_features.loc[df_features['solidity'].idxmin(), 'field_id']} "
      f"(solidity={df_features['solidity'].min():.4f})")

# Export feature matrix
df_features.to_csv("geometry_features.csv", index=False)
print("\nExported: geometry_features.csv")

print("\n=== Script 01 Complete ===")
