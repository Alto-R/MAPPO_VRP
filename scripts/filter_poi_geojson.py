#!/usr/bin/env python
"""
Filter POIs from a GeoJSON FeatureCollection by simple rules.

This repo's POI GeoJSON (e.g. data/poi_batch_1_final_[7480].geojson) only contains:
  - properties: {name, type, icon, rank, id, text_style}
So practical filtering is usually based on:
  - properties.type  (e.g. "住宅")
  - properties.name  (keyword matching for courier/express related POIs)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EARTH_RADIUS_KM = 6371.0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate the great-circle distance between two points on Earth (in km)."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def get_coords(feature: dict[str, Any]) -> tuple[float, float] | None:
    """Extract (lon, lat) from a GeoJSON Point feature."""
    geom = feature.get("geometry")
    if not geom or geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates")
    if not coords or len(coords) < 2:
        return None
    return (float(coords[0]), float(coords[1]))


DEFAULT_EXPRESS_KEYWORDS = [
    "快递",
    "快件",
    "驿站",
    "丰巢",
    "菜鸟",
    "顺丰",
    "中通",
    "圆通",
    "申通",
    "韵达",
    "京东",
    "邮政",
    "EMS",
    "速运",
    "速递",
    "快运",
    "包裹",
    "取件",
    "自提",
    "智能柜",
    "快递柜",
]

# Keep this conservative: only remove very obvious false positives by default.
DEFAULT_EXPRESS_NEG_KEYWORDS = [
    "铁骑",
]


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _contains_any(text: str, keywords: list[str]) -> bool:
    if not text or not keywords:
        return False
    lower = text.lower()
    for kw in keywords:
        if kw and kw.lower() in lower:
            return True
    return False


def _load_geojson(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError(f"Not a GeoJSON FeatureCollection: {path}")
    features = data.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON missing 'features' list: {path}")
    return data


def _write_geojson(path: Path, base: dict[str, Any], features: list[dict[str, Any]], indent: int | None) -> None:
    out = {k: v for k, v in base.items() if k != "features"}
    out["type"] = "FeatureCollection"
    out["features"] = features
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def filter_residential_by_radius(
    residential: list[dict[str, Any]],
    express: list[dict[str, Any]],
    radius_km: float,
) -> list[dict[str, Any]]:
    """Filter residential POIs that are within radius_km of any express POI."""
    express_coords = [get_coords(f) for f in express]
    express_coords = [c for c in express_coords if c is not None]

    if not express_coords:
        return []

    filtered = []
    for res in residential:
        res_coord = get_coords(res)
        if res_coord is None:
            continue
        for exp_coord in express_coords:
            dist = haversine_distance(res_coord[0], res_coord[1], exp_coord[0], exp_coord[1])
            if dist <= radius_km:
                filtered.append(res)
                break
    return filtered


def visualize_pois(
    express: list[dict[str, Any]],
    residential: list[dict[str, Any]],
    radius_km: float,
    output_path: Path,
) -> None:
    """Generate a static PNG map using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import matplotlib
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    # Set font for Chinese characters
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Collect coordinates
    exp_coords = [(get_coords(f), f.get("properties", {}).get("name", "")) for f in express]
    exp_coords = [(c, n) for c, n in exp_coords if c is not None]

    res_coords = [get_coords(f) for f in residential]
    res_coords = [c for c in res_coords if c is not None]

    if not exp_coords and not res_coords:
        print("No coordinates to visualize.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot residential points (blue)
    if res_coords:
        res_lons = [c[0] for c in res_coords]
        res_lats = [c[1] for c in res_coords]
        ax.scatter(res_lons, res_lats, c="blue", s=15, alpha=0.6, label=f"Residential ({len(res_coords)})", zorder=2)

    # Plot express points (red) with radius circles
    if exp_coords:
        # Approximate conversion: 1 degree latitude ~ 111km
        # For longitude, it varies with latitude, use average
        avg_lat = sum(c[1] for c, _ in exp_coords) / len(exp_coords) if exp_coords else 22.5
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(avg_lat))
        radius_deg_lat = radius_km / km_per_deg_lat
        radius_deg_lon = radius_km / km_per_deg_lon

        for (lon, lat), name in exp_coords:
            # Draw radius circle (ellipse due to lat/lon scaling)
            circle = Circle(
                (lon, lat),
                radius_deg_lon,  # Use longitude scale for width
                color="red",
                fill=True,
                alpha=0.15,
                zorder=1,
            )
            ax.add_patch(circle)

        exp_lons = [c[0] for c, _ in exp_coords]
        exp_lats = [c[1] for c, _ in exp_coords]
        ax.scatter(exp_lons, exp_lats, c="red", s=100, marker="^", label=f"Express ({len(exp_coords)})", zorder=3)

        # Add labels for express points
        for (lon, lat), name in exp_coords:
            ax.annotate(name, (lon, lat), fontsize=7, ha="left", va="bottom", color="darkred")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"POI Distribution (Express + Residential within {radius_km}km)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"visualization saved -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter POIs from a GeoJSON FeatureCollection.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input GeoJSON path (FeatureCollection)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="",
        help="Output directory (default: alongside input)",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output filename prefix (default: input stem)",
    )
    parser.add_argument(
        "--res-type",
        default="住宅",
        help='Residential type value to match (default: "住宅")',
    )
    parser.add_argument(
        "--res-name-kws",
        default="",
        help="Extra residential name keywords (comma-separated)",
    )
    parser.add_argument(
        "--express-kws",
        default="",
        help="Express/courier keywords (comma-separated; default: built-in list)",
    )
    parser.add_argument(
        "--express-neg-kws",
        default="",
        help="Exclude keywords (comma-separated; default: built-in list)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for output (use 0 for compact)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Filter residential within this radius (km) of express points (default: 10)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate an interactive HTML map",
    )
    parser.add_argument(
        "--combined-out",
        default="",
        help="Output filename for combined (express + filtered residential) GeoJSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem

    indent = None if args.indent == 0 else args.indent

    data = _load_geojson(input_path)
    features: list[dict[str, Any]] = data["features"]

    res_type = args.res_type
    res_name_kws = _split_csv(args.res_name_kws)

    express_kws = _split_csv(args.express_kws) or list(DEFAULT_EXPRESS_KEYWORDS)
    express_neg_kws = _split_csv(args.express_neg_kws) or list(DEFAULT_EXPRESS_NEG_KEYWORDS)

    residential: list[dict[str, Any]] = []
    express: list[dict[str, Any]] = []

    for feat in features:
        props = feat.get("properties") or {}
        name = str(props.get("name") or "")
        poi_type = str(props.get("type") or "")

        is_residential = (poi_type == res_type) or _contains_any(name, res_name_kws)
        if is_residential:
            residential.append(feat)

        is_express = _contains_any(name, express_kws) or _contains_any(poi_type, express_kws)
        is_excluded = _contains_any(name, express_neg_kws) or _contains_any(poi_type, express_neg_kws)
        if is_express and not is_excluded:
            express.append(feat)

    res_out = out_dir / f"{prefix}_residential.geojson"
    exp_out = out_dir / f"{prefix}_express.geojson"

    _write_geojson(res_out, data, residential, indent=indent)
    _write_geojson(exp_out, data, express, indent=indent)

    # Filter residential by radius from express points
    filtered_residential = filter_residential_by_radius(residential, express, args.radius)

    # Output combined GeoJSON (express + filtered residential)
    combined_out_name = args.combined_out or f"{prefix}_combined_{args.radius}km.geojson"
    combined_out = out_dir / combined_out_name
    combined_features = express + filtered_residential
    _write_geojson(combined_out, data, combined_features, indent=indent)

    # Keep console output ASCII-friendly (Windows encoding in some environments is messy).
    print(f"input_features={len(features)}")
    print(f"residential_features={len(residential)} -> {res_out}")
    print(f"express_features={len(express)} -> {exp_out}")
    print(f"filtered_residential (within {args.radius}km)={len(filtered_residential)}")
    print(f"combined_features={len(combined_features)} -> {combined_out}")

    # Visualize if requested
    if args.visualize:
        vis_out = out_dir / f"{prefix}_map_{args.radius}km.png"
        visualize_pois(express, filtered_residential, args.radius, vis_out)

    print("tip: use --express-kws / --express-neg-kws to tune keyword filtering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

