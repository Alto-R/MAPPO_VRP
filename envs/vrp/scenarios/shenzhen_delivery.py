"""
Shenzhen Real-World Delivery Scenario.

Uses real POI data from GeoJSON and OSM road network via GraphHopper.
- Depot: Express delivery point (configurable)
- Customers: Residential points within 5km radius
- Truck routes: OSM road network within 10km radius
"""

import numpy as np
import os
from mappo.envs.vrp.core import World, Truck, Drone, Customer
from mappo.envs.vrp.distance_utils import drone_distance
from mappo.envs.vrp.geo_data_loader import GeoDataLoader, CoordinateConverter


class Scenario:
    """
    Shenzhen real-world delivery scenario.

    Uses actual geographic data:
    - Express points as depot options
    - Residential points as customers
    - OSM road network for truck routing
    """

    def __init__(self):
        # Reward parameters (same as basic scenario)
        self.time_penalty = 0.1
        self.delivery_bonus = 5.0
        self.completion_bonus = 100.0
        self.incomplete_penalty = 20.0
        self.energy_cost = 0.01
        self.forced_return_penalty = 0.5

        # Geographic data (lazy loaded)
        self._geo_loader = None
        self._coord_converter = None
        self._depot_geo = None
        self._customers_geo = None
        self._road_nodes_geo = None
        self._geo_bounds = None

    def _init_geo_data(self, args):
        """Initialize geographic data loader and generate data."""
        if self._geo_loader is not None:
            return

        # Get configuration
        geojson_path = getattr(args, 'geojson_path',
            'data/poi_batch_1_final_[7480]_combined_5.0km.geojson')
        graphhopper_url = getattr(args, 'graphhopper_url', 'http://localhost:8990')
        depot_index = getattr(args, 'depot_index', 3)  # Default: 顺丰速运(福田)
        customer_radius_km = getattr(args, 'customer_radius_km', 5.0)
        road_radius_km = getattr(args, 'road_radius_km', 10.0)

        # Resolve relative path
        if not os.path.isabs(geojson_path):
            # Try relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            geojson_path = os.path.join(project_root, geojson_path)

        print(f"[ShenzhenScenario] Loading geographic data from: {geojson_path}")

        # Initialize loader
        self._geo_loader = GeoDataLoader(
            geojson_path=geojson_path,
            graphhopper_url=graphhopper_url
        )

        # Get depot
        express_points = self._geo_loader.get_express_points()
        if depot_index >= len(express_points):
            print(f"[Warning] depot_index {depot_index} out of range, using 0")
            depot_index = 0

        depot = express_points[depot_index]
        print(f"[ShenzhenScenario] Depot: {depot['name']} @ ({depot['lon']:.5f}, {depot['lat']:.5f})")

        # Snap depot to road
        depot_snapped = self._geo_loader.snap_to_road(depot['lon'], depot['lat'])
        self._depot_geo = depot_snapped
        print(f"[ShenzhenScenario] Depot snapped to road: ({depot_snapped[0]:.5f}, {depot_snapped[1]:.5f})")

        # Get customers in radius
        self._customers_geo = self._geo_loader.get_customers_in_radius(
            depot['lon'], depot['lat'], customer_radius_km
        )
        print(f"[ShenzhenScenario] Found {len(self._customers_geo)} customers within {customer_radius_km}km")

        # Generate road network nodes
        self._road_nodes_geo = self._geo_loader.generate_road_network_nodes(
            depot['lon'], depot['lat'],
            radius_km=road_radius_km,
            num_directions=8,
            points_per_direction=5
        )
        print(f"[ShenzhenScenario] Generated {len(self._road_nodes_geo)} road nodes")

        # Compute geographic bounds
        self._geo_bounds = self._geo_loader.compute_geo_bounds(
            depot['lon'], depot['lat'],
            self._road_nodes_geo,
            self._customers_geo
        )
        print(f"[ShenzhenScenario] Geo bounds: lon=[{self._geo_bounds[0]:.5f}, {self._geo_bounds[1]:.5f}], "
              f"lat=[{self._geo_bounds[2]:.5f}, {self._geo_bounds[3]:.5f}]")

        # Initialize coordinate converter
        self._coord_converter = CoordinateConverter(self._geo_bounds)

    def make_world(self, args):
        """
        Create and return the world with real geographic data.

        Args:
            args: Namespace with configuration including:
                - geojson_path: Path to POI GeoJSON file
                - depot_index: Index of express point to use as depot
                - customer_radius_km: Radius for customer selection
                - road_radius_km: Radius for road network generation
                - num_drones: Number of drones
                - num_customers: Number of customers to use (randomly selected)
                - episode_length: Max episode steps
        """
        # Initialize geographic data
        self._init_geo_data(args)

        world = World()
        world.world_length = getattr(args, 'episode_length', 200)

        # Get configuration
        num_drones = getattr(args, 'num_drones', 2)
        num_customers = getattr(args, 'num_customers', 10)

        # Thresholds - adjusted for real-world scale
        # In normalized [-1, 1] space, 0.05 is about 2.5% of the area
        world.delivery_threshold = getattr(args, 'delivery_threshold', 0.03)
        world.recovery_threshold = getattr(args, 'recovery_threshold', 0.05)

        # Create truck
        world.truck = Truck()
        world.truck.name = 'truck_0'

        # Create drones
        world.drones = []
        for i in range(num_drones):
            drone = Drone()
            drone.name = f'drone_{i}'
            drone.state.status = 'onboard'
            # Adjust battery consumption for real-world scale
            drone.battery_consumption_rate = 0.005
            world.drones.append(drone)

        # Create customers from geographic data
        world.customers = []

        # Select customers (random sample if more available than needed)
        available_customers = self._customers_geo
        if len(available_customers) > num_customers:
            indices = np.random.choice(len(available_customers), num_customers, replace=False)
            selected_customers = [available_customers[i] for i in indices]
        else:
            selected_customers = available_customers[:num_customers]

        for i, cust_geo in enumerate(selected_customers):
            customer = Customer()
            customer.name = f'customer_{i}'
            # Store geographic info for reference
            customer.geo_info = {
                'name': cust_geo['name'],
                'lon': cust_geo['lon'],
                'lat': cust_geo['lat'],
                'distance_km': cust_geo.get('distance_km', 0)
            }
            world.customers.append(customer)

        # Convert road nodes to environment coordinates
        world.route_nodes = []
        for node_geo in self._road_nodes_geo:
            env_pos = self._coord_converter.geo_to_env(node_geo[0], node_geo[1])
            world.route_nodes.append(env_pos)

        # Store geographic metadata in world for reference
        world.geo_bounds = self._geo_bounds
        world.depot_geo = self._depot_geo
        world.coord_converter = self._coord_converter

        # Initialize world
        self.reset_world(world)

        return world

    def reset_world(self, world):
        """Reset world to initial state."""
        # Reset truck to depot position
        depot_env = world.coord_converter.geo_to_env(
            world.depot_geo[0], world.depot_geo[1]
        )
        world.truck.state.p_pos = depot_env.copy()
        world.truck.state.p_geo = np.array([world.depot_geo[0], world.depot_geo[1]])
        world.truck.state.p_vel = np.zeros(world.dim_p)
        world.truck.state.current_node = 0
        world.truck.state.target_node = None
        world.truck.state.road_path_geo = None
        world.truck.action.release_drone = None
        world.truck.action.recover_drone = None
        world.truck.distance_traveled_this_step = 0.0

        # Reset drones
        for drone in world.drones:
            drone.state.p_pos = world.truck.state.p_pos.copy()
            drone.state.p_vel = np.zeros(world.dim_p)
            drone.state.battery = drone.max_battery
            drone.state.carrying_package = None
            drone.state.status = 'onboard'
            drone.state.target_pos = None
            drone.action.target_customer = None
            drone.action.return_to_truck = False
            drone.action.hover = False
            drone.battery_used_this_step = 0.0
            drone.forced_return_this_step = False

        # Reset customers with their geographic positions
        for i, customer in enumerate(world.customers):
            geo_info = customer.geo_info
            env_pos = world.coord_converter.geo_to_env(geo_info['lon'], geo_info['lat'])
            customer.state.p_pos = env_pos
            customer.state.served = False
            customer.state.demand = np.random.uniform(0.5, 1.0)

            # Time windows (in steps)
            tw_start = np.random.randint(0, world.world_length // 2)
            tw_duration = np.random.randint(world.world_length // 3, world.world_length)
            customer.state.time_window_start = tw_start
            customer.state.time_window_end = min(tw_start + tw_duration, world.world_length)
            customer.state.arrival_step = None
            customer.just_served_this_step = False

            # Color
            customer.color = np.array([0.75, 0.25, 0.25])  # Red

        world.world_step = 0

    def observation(self, agent, world):
        """Generate observation for an agent."""
        obs = []

        # Self state
        obs.extend(agent.state.p_pos)  # 2
        obs.extend(agent.state.p_vel)  # 2

        if isinstance(agent, Drone):
            # Drone-specific observation
            obs.append(agent.state.battery)  # 1
            obs.append(1.0 if agent.state.carrying_package is not None else 0.0)  # 1

            # Target position
            if agent.state.target_pos is not None:
                obs.extend(agent.state.target_pos)  # 2
            else:
                obs.extend([0.0, 0.0])  # 2

            # Onboard status
            obs.append(1.0 if agent.state.status == 'onboard' else 0.0)  # 1

            # Truck relative position
            rel_truck = world.truck.state.p_pos - agent.state.p_pos
            obs.extend(rel_truck)  # 2

            # Customer states
            for customer in world.customers:
                rel_pos = customer.state.p_pos - agent.state.p_pos
                obs.extend(rel_pos)  # 2
                obs.append(1.0 if customer.state.served else 0.0)  # 1
                obs.append(self._time_window_remaining(customer, world))  # 1
                obs.append(customer.state.demand)  # 1

            # Other drones
            for other in world.drones:
                if other is agent:
                    continue
                rel_pos = other.state.p_pos - agent.state.p_pos
                obs.extend(rel_pos)  # 2
                obs.append(other.state.battery)  # 1
                obs.append(self._encode_drone_status(other.state.status))  # 1

        else:  # Truck
            # Drones onboard mask
            for drone in world.drones:
                obs.append(1.0 if drone.state.status == 'onboard' else 0.0)

            # All drone states
            for drone in world.drones:
                rel_pos = drone.state.p_pos - agent.state.p_pos
                obs.extend(rel_pos)  # 2
                obs.extend(drone.state.p_vel)  # 2
                obs.append(drone.state.battery)  # 1
                obs.append(1.0 if drone.state.carrying_package is not None else 0.0)  # 1
                obs.append(self._encode_drone_status(drone.state.status))  # 1

            # Customer states
            for customer in world.customers:
                rel_pos = customer.state.p_pos - agent.state.p_pos
                obs.extend(rel_pos)  # 2
                obs.append(1.0 if customer.state.served else 0.0)  # 1
                obs.append(self._time_window_remaining(customer, world))  # 1
                obs.append(customer.state.demand)  # 1

        # Agent ID one-hot encoding
        agent_idx = world.policy_agents.index(agent)
        agent_id_onehot = [0.0] * len(world.policy_agents)
        agent_id_onehot[agent_idx] = 1.0
        obs.extend(agent_id_onehot)

        return np.array(obs, dtype=np.float32)

    def _time_window_remaining(self, customer, world):
        """Calculate normalized time remaining until time window closes."""
        if customer.state.served:
            return 0.0
        remaining = (customer.state.time_window_end - world.world_step) / world.world_length
        return max(0.0, min(1.0, remaining))

    def _encode_drone_status(self, status):
        """Encode drone status as a float."""
        status_map = {
            'onboard': 0.0,
            'flying': 0.25,
            'returning': 0.5,
            'crashed': 1.0
        }
        return status_map.get(status, 0.0)

    def get_share_obs(self, world):
        """Generate shared observation (global state) for centralized critic."""
        share_obs = []

        # Truck state (absolute)
        share_obs.extend(world.truck.state.p_pos)  # 2
        share_obs.extend(world.truck.state.p_vel)  # 2

        # Drone states (absolute)
        for drone in world.drones:
            share_obs.extend(drone.state.p_pos)  # 2
            share_obs.extend(drone.state.p_vel)  # 2
            share_obs.append(drone.state.battery)  # 1
            share_obs.append(1.0 if drone.state.carrying_package is not None else 0.0)  # 1
            share_obs.append(self._encode_drone_status(drone.state.status))  # 1

        # Customer states (absolute)
        for customer in world.customers:
            share_obs.extend(customer.state.p_pos)  # 2
            share_obs.append(1.0 if customer.state.served else 0.0)  # 1
            share_obs.append(self._time_window_remaining(customer, world))  # 1
            share_obs.append(customer.state.demand)  # 1

        # Normalized time step
        share_obs.append(world.world_step / world.world_length)  # 1

        return np.array(share_obs, dtype=np.float32)

    def compute_share_obs_dim(self, world):
        """Compute the dimension of shared observation."""
        return 4 + len(world.drones) * 7 + len(world.customers) * 5 + 1

    def get_available_actions(self, agent, world):
        """Return available actions mask."""
        if isinstance(agent, Drone):
            mask = np.ones(2 + len(world.customers))

            if agent.state.status == 'onboard':
                mask[1:] = 0  # Only HOVER
            elif agent.state.status == 'crashed':
                mask[:] = 0
                mask[0] = 1  # Only HOVER
            else:
                # Disable served customers
                for i, c in enumerate(world.customers):
                    if c.state.served:
                        mask[2 + i] = 0

                # If carrying, must deliver to target
                if agent.state.carrying_package is not None:
                    mask[1] = 0  # No return
                    for i in range(len(world.customers)):
                        if i != agent.state.carrying_package:
                            mask[2 + i] = 0

        else:  # Truck
            num_nodes = len(world.route_nodes)
            num_drones = len(world.drones)
            mask = np.ones(1 + num_nodes + 2 * num_drones)

            # RELEASE: only for onboard drones
            for i, d in enumerate(world.drones):
                if d.state.status != 'onboard':
                    mask[1 + num_nodes + i] = 0

            # RECOVER: only for nearby non-onboard drones
            for i, d in enumerate(world.drones):
                dist = drone_distance(d.state.p_pos, agent.state.p_pos)
                if dist > world.recovery_threshold or d.state.status == 'onboard':
                    mask[1 + num_nodes + num_drones + i] = 0

        return mask

    def is_terminal(self, world):
        """Check if episode should terminate."""
        if world.world_step >= world.world_length:
            return True
        if all(c.state.served for c in world.customers):
            return True
        if all(d.state.status == 'crashed' for d in world.drones):
            return True
        return False

    def compute_global_reward(self, world):
        """Compute global reward for cooperative scenario."""
        rew = 0.0

        # Time penalty
        rew -= self.time_penalty

        # Delivery rewards
        for c in world.customers:
            if c.just_served_this_step:
                rew += self.delivery_bonus

        # Energy cost
        for d in world.drones:
            rew -= self.energy_cost * d.battery_used_this_step

        # Forced return penalty
        for d in world.drones:
            if d.forced_return_this_step:
                rew -= self.forced_return_penalty

        # Terminal reward
        if self.is_terminal(world):
            served = sum(1 for c in world.customers if c.state.served)
            total = len(world.customers)
            if served == total:
                rew += self.completion_bonus
            else:
                rew -= self.incomplete_penalty * (total - served)

        return rew

    def info(self, agent, world):
        """Return info dict for an agent."""
        return {
            'customers_served': sum(1 for c in world.customers if c.state.served),
            'total_customers': len(world.customers),
            'time_step': world.world_step,
        }
