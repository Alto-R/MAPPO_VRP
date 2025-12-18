"""
Geographic data loader for real-world VRP scenarios.

Loads POI data from GeoJSON and generates truck route nodes from OSM road network.
"""

import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class GeoDataLoader:
    """
    Loads geographic data and generates route nodes from road network.
    """

    # Express-related keywords for filtering delivery points
    EXPRESS_KEYWORDS = ['快递', '快运', '速运', '邮政', '驿站', '菜鸟', '丰巢', '快件']

    def __init__(
        self,
        geojson_path: str,
        graphhopper_url: str = "http://localhost:8990"
    ):
        """
        Initialize the data loader.

        Args:
            geojson_path: Path to GeoJSON file containing POI data
            graphhopper_url: GraphHopper service URL
        """
        self.geojson_path = geojson_path
        self.graphhopper_url = graphhopper_url

        # Load GeoJSON data
        self._features = []
        self._express_points = []
        self._residential_points = []
        self._load_geojson()

        # GraphHopper client (lazy initialization)
        self._gh_client = None

    def _load_geojson(self):
        """Load and parse GeoJSON file."""
        if not os.path.exists(self.geojson_path):
            raise FileNotFoundError(f"GeoJSON file not found: {self.geojson_path}")

        with open(self.geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self._features = data.get('features', [])

        # Classify points
        for i, feat in enumerate(self._features):
            props = feat.get('properties', {})
            name = props.get('name', '')
            poi_type = props.get('type', '')
            coords = feat.get('geometry', {}).get('coordinates', [0, 0])

            point_data = {
                'index': i,
                'name': name,
                'type': poi_type,
                'lon': coords[0],
                'lat': coords[1]
            }

            # Check if express-related
            if any(kw in name for kw in self.EXPRESS_KEYWORDS):
                self._express_points.append(point_data)

            # Check if residential
            if poi_type == '住宅':
                self._residential_points.append(point_data)

        print(f"[GeoDataLoader] Loaded {len(self._features)} features")
        print(f"[GeoDataLoader] Found {len(self._express_points)} express points")
        print(f"[GeoDataLoader] Found {len(self._residential_points)} residential points")

    def _get_gh_client(self):
        """Get or create GraphHopper client."""
        if self._gh_client is None:
            from mappo.tools.graphhopper.gh_client import GraphHopperClient
            self._gh_client = GraphHopperClient(base_url=self.graphhopper_url)
            if not self._gh_client.is_available():
                raise ConnectionError(
                    f"GraphHopper service not available at {self.graphhopper_url}"
                )
        return self._gh_client

    def get_express_points(self) -> List[Dict]:
        """Return list of express/delivery points."""
        return self._express_points.copy()

    def get_residential_points(self) -> List[Dict]:
        """Return list of residential points."""
        return self._residential_points.copy()

    def get_customers_in_radius(
        self,
        center_lon: float,
        center_lat: float,
        radius_km: float = 5.0
    ) -> List[Dict]:
        """
        Get residential points within specified radius.

        Args:
            center_lon: Center longitude
            center_lat: Center latitude
            radius_km: Radius in kilometers

        Returns:
            List of residential points within radius, sorted by distance
        """
        result = []
        for point in self._residential_points:
            dist = self._haversine_km(
                center_lon, center_lat,
                point['lon'], point['lat']
            )
            if dist <= radius_km:
                point_copy = point.copy()
                point_copy['distance_km'] = dist
                result.append(point_copy)

        # Sort by distance
        result.sort(key=lambda x: x['distance_km'])
        return result

    def snap_to_road(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Snap a point to the nearest road using GraphHopper.

        Method: Request a very short route from the point to itself,
        GraphHopper will snap both endpoints to the nearest road.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Snapped coordinates (lon, lat)
        """
        gh = self._get_gh_client()

        # Request route to a very close point
        # GraphHopper will snap to road
        offset = 0.0001  # ~10 meters
        try:
            route = gh.route(
                start=(lon, lat),
                end=(lon + offset, lat + offset),
                profile="truck",
                calc_points=True
            )
            if route['points'] and len(route['points']) > 0:
                # First point is the snapped start location
                snapped = route['points'][0]
                return (snapped[0], snapped[1])
        except Exception as e:
            print(f"[GeoDataLoader] Snap to road failed: {e}, using original point")

        return (lon, lat)

    def generate_road_network_nodes(
        self,
        center_lon: float,
        center_lat: float,
        radius_km: float = 10.0,
        num_directions: int = 8,
        points_per_direction: int = 5
    ) -> List[Tuple[float, float]]:
        """
        Generate route nodes by sampling from OSM road network.

        Method: From center, request routes to multiple directions,
        extract path points as drivable nodes.

        Args:
            center_lon: Center longitude (depot location)
            center_lat: Center latitude (depot location)
            radius_km: Maximum radius to explore (km)
            num_directions: Number of directions to explore (8 = N,NE,E,SE,S,SW,W,NW)
            points_per_direction: Number of points to sample per direction

        Returns:
            List of (lon, lat) coordinates on road network
        """
        gh = self._get_gh_client()
        all_road_points = []

        # Snap center to road first
        center_snapped = self.snap_to_road(center_lon, center_lat)
        all_road_points.append(center_snapped)

        # Generate target points in each direction
        for i in range(num_directions):
            angle = 2 * math.pi * i / num_directions

            # Target point at radius_km in this direction
            target_lon, target_lat = self._offset_point(
                center_lon, center_lat, radius_km, angle
            )

            try:
                # Get route to target
                route = gh.route(
                    start=center_snapped,
                    end=(target_lon, target_lat),
                    profile="truck",
                    calc_points=True
                )

                path_points = route.get('points', [])
                if path_points:
                    # Sample points along the route
                    num_points = len(path_points)
                    if num_points > points_per_direction:
                        # Sample evenly
                        indices = np.linspace(0, num_points - 1, points_per_direction, dtype=int)
                        for idx in indices:
                            pt = path_points[idx]
                            all_road_points.append((pt[0], pt[1]))
                    else:
                        # Use all points
                        for pt in path_points:
                            all_road_points.append((pt[0], pt[1]))

            except Exception as e:
                print(f"[GeoDataLoader] Route to direction {i} failed: {e}")
                continue

        # Remove duplicates (within small tolerance)
        unique_points = self._remove_duplicate_points(all_road_points, tolerance_m=50)

        print(f"[GeoDataLoader] Generated {len(unique_points)} road network nodes")
        return unique_points

    def _offset_point(
        self,
        lon: float,
        lat: float,
        distance_km: float,
        bearing_rad: float
    ) -> Tuple[float, float]:
        """
        Calculate a new point at given distance and bearing from origin.

        Args:
            lon, lat: Origin coordinates
            distance_km: Distance in kilometers
            bearing_rad: Bearing in radians (0 = North, pi/2 = East)

        Returns:
            New (lon, lat)
        """
        # Approximate conversion
        lat_offset = distance_km / 110.574 * math.cos(bearing_rad)
        lon_offset = distance_km / (111.320 * math.cos(math.radians(lat))) * math.sin(bearing_rad)

        return (lon + lon_offset, lat + lat_offset)

    def _remove_duplicate_points(
        self,
        points: List[Tuple[float, float]],
        tolerance_m: float = 50
    ) -> List[Tuple[float, float]]:
        """Remove points that are too close to each other."""
        if not points:
            return []

        unique = [points[0]]
        tolerance_km = tolerance_m / 1000

        for pt in points[1:]:
            is_duplicate = False
            for existing in unique:
                dist = self._haversine_km(pt[0], pt[1], existing[0], existing[1])
                if dist < tolerance_km:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(pt)

        return unique

    @staticmethod
    def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate haversine distance in kilometers."""
        R = 6371  # Earth radius in km

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def compute_geo_bounds(
        self,
        center_lon: float,
        center_lat: float,
        points: List[Tuple[float, float]],
        customers: List[Dict],
        padding: float = 0.01
    ) -> Tuple[float, float, float, float]:
        """
        Compute geographic bounds that include all points.

        Args:
            center_lon, center_lat: Center coordinates
            points: Road network nodes
            customers: Customer points
            padding: Padding in degrees

        Returns:
            (min_lon, max_lon, min_lat, max_lat)
        """
        all_lons = [center_lon] + [p[0] for p in points] + [c['lon'] for c in customers]
        all_lats = [center_lat] + [p[1] for p in points] + [c['lat'] for c in customers]

        min_lon = min(all_lons) - padding
        max_lon = max(all_lons) + padding
        min_lat = min(all_lats) - padding
        max_lat = max(all_lats) + padding

        return (min_lon, max_lon, min_lat, max_lat)


class CoordinateConverter:
    """
    Converts between geographic coordinates and environment coordinates.
    """

    def __init__(
        self,
        geo_bounds: Tuple[float, float, float, float],
        env_bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Initialize converter.

        Args:
            geo_bounds: (min_lon, max_lon, min_lat, max_lat)
            env_bounds: (min, max) for environment coordinates
        """
        self.min_lon, self.max_lon, self.min_lat, self.max_lat = geo_bounds
        self.env_min, self.env_max = env_bounds

        # Pre-calculate ranges
        self.lon_range = self.max_lon - self.min_lon
        self.lat_range = self.max_lat - self.min_lat
        self.env_range = self.env_max - self.env_min

    def geo_to_env(self, lon: float, lat: float) -> np.ndarray:
        """Convert geographic to environment coordinates."""
        # Normalize to [0, 1]
        norm_lon = (lon - self.min_lon) / self.lon_range
        norm_lat = (lat - self.min_lat) / self.lat_range

        # Map to env_bounds
        x = self.env_min + norm_lon * self.env_range
        y = self.env_min + norm_lat * self.env_range

        return np.array([x, y], dtype=np.float32)

    def env_to_geo(self, pos: np.ndarray) -> Tuple[float, float]:
        """Convert environment to geographic coordinates."""
        # Normalize from env_bounds to [0, 1]
        norm_x = (pos[0] - self.env_min) / self.env_range
        norm_y = (pos[1] - self.env_min) / self.env_range

        # Map to geographic
        lon = self.min_lon + norm_x * self.lon_range
        lat = self.min_lat + norm_y * self.lat_range

        return (lon, lat)

    def geo_to_env_batch(self, points: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Convert multiple geographic points to environment coordinates."""
        return [self.geo_to_env(lon, lat) for lon, lat in points]


if __name__ == "__main__":
    # Test the loader
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    loader = GeoDataLoader(
        geojson_path="data/poi_batch_1_final_[7480]_combined_5.0km.geojson"
    )

    # Get express points
    express = loader.get_express_points()
    print("\n=== Express Points ===")
    for i, p in enumerate(express):
        print(f"{i}: {p['name']} @ ({p['lon']:.5f}, {p['lat']:.5f})")

    # Select depot (index 3 = 顺丰速运福田)
    depot = express[3]
    print(f"\n=== Selected Depot ===")
    print(f"{depot['name']} @ ({depot['lon']:.5f}, {depot['lat']:.5f})")

    # Get customers in 5km
    customers = loader.get_customers_in_radius(depot['lon'], depot['lat'], 5.0)
    print(f"\n=== Customers within 5km: {len(customers)} ===")
    for c in customers[:5]:
        print(f"  {c['name']} - {c['distance_km']:.2f}km")

    # Generate road nodes (requires GraphHopper)
    try:
        nodes = loader.generate_road_network_nodes(
            depot['lon'], depot['lat'],
            radius_km=10.0,
            num_directions=8,
            points_per_direction=5
        )
        print(f"\n=== Road Network Nodes: {len(nodes)} ===")
        for i, n in enumerate(nodes[:10]):
            print(f"  {i}: ({n[0]:.5f}, {n[1]:.5f})")
    except Exception as e:
        print(f"\n[Warning] Could not generate road nodes: {e}")
