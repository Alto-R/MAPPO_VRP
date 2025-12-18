"""
Distance calculation utilities for VRP environment.

Provides two distance calculation modes:
1. Drone: L2 norm (Euclidean distance) - straight line flight
2. Truck: GraphHopper road network distance - real road routing
"""

import numpy as np
from typing import Tuple, Optional, Dict
from functools import lru_cache


class DistanceCalculator:
    """
    Distance calculator with support for different modes.

    - Drone distances: Always use L2 norm (direct flight)
    - Truck distances: Use GraphHopper API for road network distance,
                       fallback to L2 norm if unavailable
    """

    def __init__(
        self,
        use_graphhopper: bool = True,
        graphhopper_url: str = "http://localhost:8990",
        coord_bounds: Tuple[float, float, float, float] = None,
        geo_bounds: Tuple[float, float, float, float] = None,
    ):
        """
        Initialize distance calculator.

        Args:
            use_graphhopper: Whether to use GraphHopper for truck distances
            graphhopper_url: GraphHopper service URL
            coord_bounds: Environment coordinate bounds (min_x, max_x, min_y, max_y)
            geo_bounds: Geographic bounds for coordinate conversion (min_lon, max_lon, min_lat, max_lat)
        """
        self.use_graphhopper = use_graphhopper
        self.graphhopper_url = graphhopper_url
        self.gh_client = None
        self.gh_available = False

        # Coordinate conversion bounds
        # Default: environment uses [-1, 1] normalized coords
        self.coord_bounds = coord_bounds or (-1.0, 1.0, -1.0, 1.0)
        # Default: Guangzhou area geographic bounds
        self.geo_bounds = geo_bounds or (113.1, 113.5, 22.9, 23.3)

        # Distance cache to avoid repeated API calls
        self._road_distance_cache: Dict[Tuple[Tuple[float, float], Tuple[float, float]], float] = {}

        if use_graphhopper:
            self._init_graphhopper()

    def _init_graphhopper(self):
        """Initialize GraphHopper client. Raises error if service not available."""
        try:
            from mappo.tools.graphhopper.gh_client import GraphHopperClient
            self.gh_client = GraphHopperClient(base_url=self.graphhopper_url)
            self.gh_available = self.gh_client.is_available()
            if self.gh_available:
                print("[DistanceCalculator] GraphHopper service connected")
            else:
                raise ConnectionError(
                    f"[DistanceCalculator] GraphHopper service not available at {self.graphhopper_url}. "
                    "Please start GraphHopper server first, or set use_graphhopper=False."
                )
        except ImportError:
            raise ImportError(
                "[DistanceCalculator] GraphHopper client not found. "
                "Please ensure mappo/tools/graphhopper/gh_client.py exists."
            )
        except ConnectionError:
            raise  # Re-raise ConnectionError
        except Exception as e:
            raise RuntimeError(f"[DistanceCalculator] GraphHopper init error: {e}")

    def env_to_geo(self, pos: np.ndarray) -> Tuple[float, float]:
        """
        Convert environment coordinates to geographic coordinates (lon, lat).

        Args:
            pos: Environment position [x, y] in coord_bounds range

        Returns:
            Geographic coordinates (longitude, latitude)
        """
        min_x, max_x, min_y, max_y = self.coord_bounds
        min_lon, max_lon, min_lat, max_lat = self.geo_bounds

        # Normalize to [0, 1]
        norm_x = (pos[0] - min_x) / (max_x - min_x)
        norm_y = (pos[1] - min_y) / (max_y - min_y)

        # Map to geographic coordinates
        lon = min_lon + norm_x * (max_lon - min_lon)
        lat = min_lat + norm_y * (max_lat - min_lat)

        return (lon, lat)

    def geo_to_env(self, lon: float, lat: float) -> np.ndarray:
        """
        Convert geographic coordinates to environment coordinates.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Environment position [x, y]
        """
        min_x, max_x, min_y, max_y = self.coord_bounds
        min_lon, max_lon, min_lat, max_lat = self.geo_bounds

        # Normalize geographic to [0, 1]
        norm_lon = (lon - min_lon) / (max_lon - min_lon)
        norm_lat = (lat - min_lat) / (max_lat - min_lat)

        # Map to environment coordinates
        x = min_x + norm_lon * (max_x - min_x)
        y = min_y + norm_lat * (max_y - min_y)

        return np.array([x, y])

    def drone_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate drone distance (L2 norm / Euclidean distance).
        Drones fly in straight lines.

        Args:
            pos1: First position [x, y]
            pos2: Second position [x, y]

        Returns:
            Euclidean distance
        """
        return np.linalg.norm(pos1 - pos2)

    def truck_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate truck distance using road network (GraphHopper) or L2 fallback.

        Args:
            pos1: First position [x, y] in environment coordinates
            pos2: Second position [x, y] in environment coordinates

        Returns:
            Road network distance (in environment coordinate units),
            or L2 distance if GraphHopper unavailable
        """
        # If same position, return 0
        if np.allclose(pos1, pos2, atol=1e-6):
            return 0.0

        # If GraphHopper not available, use L2 fallback
        if not self.use_graphhopper or not self.gh_available:
            return self.drone_distance(pos1, pos2)

        # Create cache key (rounded to avoid floating point issues)
        key = (
            (round(pos1[0], 4), round(pos1[1], 4)),
            (round(pos2[0], 4), round(pos2[1], 4))
        )

        # Check cache
        if key in self._road_distance_cache:
            return self._road_distance_cache[key]

        # Also check reverse direction (road distance is symmetric for our purposes)
        reverse_key = (key[1], key[0])
        if reverse_key in self._road_distance_cache:
            return self._road_distance_cache[reverse_key]

        try:
            # Convert to geographic coordinates
            geo1 = self.env_to_geo(pos1)
            geo2 = self.env_to_geo(pos2)

            # Call GraphHopper API
            route = self.gh_client.route(
                start=geo1,
                end=geo2,
                profile="truck",
                calc_points=False,  # Don't need path points, just distance
                instructions=False
            )

            # Get road distance in meters
            road_distance_meters = route['distance']

            # Convert meters to environment coordinate units
            # Calculate the scale factor based on geographic extent
            env_distance = self._meters_to_env_units(road_distance_meters)

            # Cache the result
            self._road_distance_cache[key] = env_distance

            return env_distance

        except Exception as e:
            # On error, raise exception instead of silent fallback
            raise RuntimeError(
                f"[DistanceCalculator] GraphHopper route calculation failed: {e}. "
                f"From {pos1} to {pos2} (geo: {self.env_to_geo(pos1)} to {self.env_to_geo(pos2)})"
            )

    def _meters_to_env_units(self, meters: float) -> float:
        """
        Convert distance in meters to environment coordinate units.

        This uses an approximation based on the geographic bounds:
        - Calculate the diagonal distance of the geographic area in meters
        - Calculate the diagonal distance of the environment coordinate space
        - Use the ratio to convert
        """
        min_lon, max_lon, min_lat, max_lat = self.geo_bounds
        min_x, max_x, min_y, max_y = self.coord_bounds

        # Approximate geographic diagonal in meters
        # Using simple equirectangular approximation
        lat_mid = (min_lat + max_lat) / 2
        lon_scale = 111320 * np.cos(np.radians(lat_mid))  # meters per degree longitude
        lat_scale = 110540  # meters per degree latitude (approximate)

        geo_width_m = (max_lon - min_lon) * lon_scale
        geo_height_m = (max_lat - min_lat) * lat_scale
        geo_diagonal_m = np.sqrt(geo_width_m**2 + geo_height_m**2)

        # Environment coordinate diagonal
        env_width = max_x - min_x
        env_height = max_y - min_y
        env_diagonal = np.sqrt(env_width**2 + env_height**2)

        # Scale factor: env_units per meter
        scale = env_diagonal / geo_diagonal_m

        return meters * scale

    def clear_cache(self):
        """Clear the road distance cache."""
        self._road_distance_cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cache_size': len(self._road_distance_cache),
            'graphhopper_available': self.gh_available
        }

    def get_truck_route_path(
        self,
        pos1: np.ndarray,
        pos2: np.ndarray,
        simplify: bool = True,
        max_points: int = 50
    ) -> Optional[list]:
        """
        Get the actual road path between two positions.

        Args:
            pos1: Start position [x, y] in environment coordinates
            pos2: End position [x, y] in environment coordinates
            simplify: Whether to simplify the path
            max_points: Maximum number of points to return

        Returns:
            List of points [[lon, lat], ...] representing the road path,
            or None if GraphHopper unavailable
        """
        if not self.use_graphhopper or not self.gh_available:
            return None

        # Same position - no route needed
        if np.allclose(pos1, pos2, atol=1e-6):
            geo = self.env_to_geo(pos1)
            return [[geo[0], geo[1]]]

        try:
            # Convert to geographic coordinates
            geo1 = self.env_to_geo(pos1)
            geo2 = self.env_to_geo(pos2)

            # Call GraphHopper API with calc_points=True
            route = self.gh_client.route(
                start=geo1,
                end=geo2,
                profile="truck",
                calc_points=True,
                points_encoded=False,
                instructions=False
            )

            # Extract path points (gh_client already extracts coordinates as a list)
            points = route.get('points', [])

            if not points:
                # Fallback to straight line
                return [[geo1[0], geo1[1]], [geo2[0], geo2[1]]]

            # Simplify path if requested
            if simplify and len(points) > max_points:
                original_end = points[-1]
                step = len(points) // max_points
                points = points[::step]
                # Ensure end point is included
                if points[-1] != original_end:
                    points.append(original_end)

            return points

        except Exception as e:
            print(f"[DistanceCalculator] Route path query failed: {e}")
            # Fallback to straight line
            geo1 = self.env_to_geo(pos1)
            geo2 = self.env_to_geo(pos2)
            return [[geo1[0], geo1[1]], [geo2[0], geo2[1]]]


# Global instance for convenience
_default_calculator: Optional[DistanceCalculator] = None


def get_distance_calculator(
    use_graphhopper: bool = True,
    graphhopper_url: str = "http://localhost:8990",
    coord_bounds: Tuple[float, float, float, float] = None,
    geo_bounds: Tuple[float, float, float, float] = None,
    force_new: bool = False
) -> DistanceCalculator:
    """
    Get or create the default distance calculator instance.

    Args:
        use_graphhopper: Whether to use GraphHopper for truck distances
        graphhopper_url: GraphHopper service URL
        coord_bounds: Environment coordinate bounds
        geo_bounds: Geographic bounds for coordinate conversion
        force_new: Force creation of a new instance

    Returns:
        DistanceCalculator instance
    """
    global _default_calculator

    if _default_calculator is None or force_new:
        _default_calculator = DistanceCalculator(
            use_graphhopper=use_graphhopper,
            graphhopper_url=graphhopper_url,
            coord_bounds=coord_bounds,
            geo_bounds=geo_bounds
        )

    return _default_calculator


def drone_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Convenience function for drone distance calculation (L2 norm).
    """
    return np.linalg.norm(pos1 - pos2)


def truck_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Convenience function for truck distance calculation.
    Uses the default calculator if available, otherwise L2.
    """
    if _default_calculator is not None:
        return _default_calculator.truck_distance(pos1, pos2)
    return np.linalg.norm(pos1 - pos2)
