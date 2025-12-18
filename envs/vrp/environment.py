"""
Multi-Agent VRP Environment for MAPPO training.
Implements Gym-style interface compatible with on-policy MAPPO framework.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mappo.envs.vrp.core import Truck, Drone
from mappo.envs.vrp.distance_utils import DistanceCalculator, drone_distance


class MultiAgentVRPEnv(gym.Env):
    """
    Multi-Agent VRP Environment.

    Agent ordering is FIXED: [truck, drone_0, drone_1, ...]
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, available_actions_callback=None,
                 share_obs_callback=None, discrete_action=True,
                 use_graphhopper: bool = True,
                 graphhopper_url: str = "http://localhost:8995",
                 geo_bounds: tuple = None):
        """
        Initialize the environment.

        Args:
            world: World object containing all entities
            reset_callback: Function to reset world state
            reward_callback: Function to compute reward (not used, use global reward)
            observation_callback: Function to generate observation for agent
            info_callback: Function to generate info dict
            done_callback: Function to check terminal condition
            available_actions_callback: Function to get action mask
            share_obs_callback: Function to get shared observation
            discrete_action: Whether to use discrete action space
            use_graphhopper: Whether to use GraphHopper for truck distance calculation
            graphhopper_url: GraphHopper service URL
            geo_bounds: Geographic bounds (min_lon, max_lon, min_lat, max_lat) for coordinate conversion
        """
        self.world = world
        self.world_length = world.world_length
        self.current_step = 0

        # Agents in FIXED order: [truck, drone_0, drone_1, ...]
        self.agents = world.policy_agents
        self.n = len(self.agents)

        # Callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.available_actions_callback = available_actions_callback
        self.share_obs_callback = share_obs_callback

        # Settings
        self.discrete_action_space = discrete_action
        self.shared_reward = True  # Cooperative scenario

        # Initialize distance calculator
        # Truck uses GraphHopper for road network distance, drone uses L2 norm
        coord_bounds = (world.bounds[0], world.bounds[1], world.bounds[0], world.bounds[1])
        # Use geo_bounds from world if available (set by scenario like shenzhen_delivery)
        actual_geo_bounds = getattr(world, 'geo_bounds', None) or geo_bounds
        self.distance_calculator = DistanceCalculator(
            use_graphhopper=use_graphhopper,
            graphhopper_url=graphhopper_url,
            coord_bounds=coord_bounds,
            geo_bounds=actual_geo_bounds
        )

        # Configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # Calculate dimensions
        self._setup_spaces()

    def _setup_spaces(self):
        """Set up action and observation spaces for all agents."""
        share_obs_dim = 0

        # First pass: calculate raw observation dimensions
        raw_obs_dims = []
        for i, agent in enumerate(self.agents):
            # Action space
            if isinstance(agent, Truck):
                # [STAY, MOVE_0..N, RELEASE_0..D, RECOVER_0..D]
                num_nodes = len(self.world.route_nodes)
                num_drones = len(self.world.drones)
                act_dim = 1 + num_nodes + 2 * num_drones
            else:  # Drone
                # [HOVER, RETURN, DELIVER_0..C]
                act_dim = 2 + len(self.world.customers)

            self.action_space.append(spaces.Discrete(act_dim))

            # Observation dimension
            obs = self.observation_callback(agent, self.world)
            raw_obs_dims.append(len(obs))

        # Pad all observations to max dimension for compatibility with vectorized envs
        self.max_obs_dim = max(raw_obs_dims)
        self.raw_obs_dims = raw_obs_dims  # Store for later use

        for i in range(self.n):
            self.observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.max_obs_dim,), dtype=np.float32)
            )

        # Shared observation space (same dimension for all agents)
        if self.share_obs_callback is not None:
            share_obs = self.share_obs_callback(self.world)
            share_obs_dim = len(share_obs)
        else:
            # Fallback: concatenate all observations
            share_obs_dim = sum(raw_obs_dims)

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.n)
        ]

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)

    def reset(self):
        """
        Reset the environment.

        Returns:
            obs_n: List of observations, one per agent
        """
        self.current_step = 0

        # Reset world
        if self.reset_callback is not None:
            self.reset_callback(self.world)

        # Reset world step counter
        self.world.world_step = 0

        # Collect observations
        obs_n = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    def step(self, action_n):
        """
        Execute one environment step.

        Args:
            action_n: List of actions, one per agent in order [truck, drone_0, drone_1, ...]

        Returns:
            obs_n: List of observations
            reward_n: List of rewards [[r], [r], ...] format
            done_n: List of done flags (all same for cooperative)
            info_n: List of info dicts
        """
        self.current_step += 1

        # 0. Reset per-step flags
        self._reset_step_flags()

        # 1. Parse actions
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        # 2. Enforce battery constraints (may override drone actions)
        self._enforce_battery_constraints()

        # 3. Process truck release/recover
        self._process_truck_release_recover()

        # 4. Update drone states (movement, battery)
        self._update_drone_states()

        # 5. Update truck state (movement)
        self._update_truck_state()

        # 6. Check deliveries
        self._check_deliveries()

        # 7. Advance world step
        self.world.step()

        # 8. Compute global reward (ONCE!)
        global_reward = self._compute_global_reward()

        # 9. Collect observations, rewards, dones, infos
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []

        # Get shared observation (same for all agents)
        share_obs = self._get_share_obs()

        # Check terminal condition (same for all agents)
        is_done = self._is_terminal()

        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([global_reward])  # All agents get same reward
            done_n.append(is_done)  # Synchronized done

            info = self._get_info(agent)
            info['available_actions'] = self._get_available_actions(agent)
            info['policy_id'] = 0 if i == 0 else 1  # truck=0, drone=1
            info['share_obs'] = share_obs
            info_n.append(info)

        # 10. Clear per-step flags
        self._clear_step_flags()

        return obs_n, reward_n, done_n, info_n

    def _reset_step_flags(self):
        """Reset per-step tracking flags."""
        for drone in self.world.drones:
            drone.battery_used_this_step = 0.0
            drone.forced_return_this_step = False
        self.world.truck.distance_traveled_this_step = 0.0

    def _clear_step_flags(self):
        """Clear per-step flags after reward calculation."""
        for customer in self.world.customers:
            customer.just_served_this_step = False

    def _set_action(self, action, agent, action_space):
        """
        Parse discrete action into agent action commands.

        Args:
            action: Discrete action index (int) or one-hot encoded action (array)
            agent: The agent (Truck or Drone)
            action_space: The action space for this agent
        """
        # Convert one-hot to discrete index if needed
        if hasattr(action, '__len__') and len(action) > 1:
            # One-hot encoded action - find the index of max value
            action = int(np.argmax(action))

        if isinstance(agent, Drone):
            # Reset action
            agent.action.hover = False
            agent.action.return_to_truck = False
            agent.action.target_customer = None

            if action == 0:  # HOVER
                agent.action.hover = True
            elif action == 1:  # RETURN_TO_TRUCK
                agent.action.return_to_truck = True
            else:  # DELIVER_TO_CUSTOMER_X
                customer_idx = action - 2
                if customer_idx < len(self.world.customers):
                    agent.action.target_customer = customer_idx

        else:  # Truck
            # Reset per-step actions (release/recover are one-shot)
            agent.action.release_drone = None
            agent.action.recover_drone = None

            num_nodes = len(self.world.route_nodes)
            num_drones = len(self.world.drones)

            if action == 0:  # STAY
                # Clear target - stop moving
                agent.state.target_node = None
                agent.state.road_path = None
                agent.state.path_index = 0
            elif action < 1 + num_nodes:  # MOVE_TO_NODE_X
                # Set new target node (persistent until reached or changed)
                node_idx = action - 1
                # Clear path cache if target changed (will be re-fetched)
                if agent.state.target_node != node_idx:
                    agent.state.road_path = None
                    agent.state.path_index = 0
                agent.state.target_node = node_idx
            elif action < 1 + num_nodes + num_drones:  # RELEASE_DRONE_X
                # Release drone - doesn't affect movement
                drone_idx = action - 1 - num_nodes
                agent.action.release_drone = drone_idx
            else:  # RECOVER_DRONE_X
                # Recover drone - doesn't affect movement
                drone_idx = action - 1 - num_nodes - num_drones
                agent.action.recover_drone = drone_idx

    def _enforce_battery_constraints(self):
        """Enforce battery constraints: force return if battery too low."""
        for drone in self.world.drones:
            if drone.state.status == 'onboard' or drone.state.status == 'crashed':
                continue

            # Calculate battery needed to return to truck (current position)
            # Drone uses L2 (straight-line) distance
            dist_to_truck = drone_distance(
                drone.state.p_pos, self.world.truck.state.p_pos
            )
            battery_needed = dist_to_truck * drone.battery_consumption_rate * 1.2  # 20% safety margin

            if drone.state.battery < battery_needed:
                # Override action: force return
                drone.action.return_to_truck = True
                drone.action.target_customer = None
                drone.action.hover = False
                drone.forced_return_this_step = True

    def _process_truck_release_recover(self):
        """Process truck's release and recover actions."""
        truck = self.world.truck

        # Release drone
        if truck.action.release_drone is not None:
            drone_idx = truck.action.release_drone
            if 0 <= drone_idx < len(self.world.drones):
                drone = self.world.drones[drone_idx]
                if drone.state.status == 'onboard':
                    drone.state.status = 'flying'
                    # Drone starts at truck position
                    drone.state.p_pos = truck.state.p_pos.copy()

        # Recover drone
        if truck.action.recover_drone is not None:
            drone_idx = truck.action.recover_drone
            if 0 <= drone_idx < len(self.world.drones):
                drone = self.world.drones[drone_idx]
                # Drone distance (L2) for recovery check
                dist = drone_distance(drone.state.p_pos, truck.state.p_pos)
                if dist <= self.world.recovery_threshold and drone.state.status != 'onboard':
                    drone.state.status = 'onboard'
                    drone.state.p_pos = truck.state.p_pos.copy()
                    drone.state.p_vel = np.zeros(self.world.dim_p)
                    drone.state.target_pos = None
                    drone.action.target_customer = None
                    drone.action.return_to_truck = False
                    # Recharge battery when recovered
                    drone.state.battery = min(1.0, drone.state.battery + 0.2)

    def _update_drone_states(self):
        """Update drone positions and battery."""
        for drone in self.world.drones:
            if drone.state.status == 'onboard':
                # Follows truck
                drone.state.p_pos = self.world.truck.state.p_pos.copy()
                drone.state.p_vel = np.zeros(self.world.dim_p)
                continue

            if drone.state.status == 'crashed':
                # No movement
                drone.state.p_vel = np.zeros(self.world.dim_p)
                continue

            # Determine target position
            target = None
            if drone.action.hover:
                # No movement
                drone.state.p_vel = np.zeros(self.world.dim_p)
                drone.state.battery -= 0.001  # Small hover cost
                drone.battery_used_this_step += 0.001
                continue
            elif drone.action.return_to_truck:
                target = self.world.truck.state.p_pos
                drone.state.status = 'returning'
            elif drone.action.target_customer is not None:
                customer = self.world.customers[drone.action.target_customer]
                target = customer.state.p_pos
                drone.state.target_pos = target.copy()
                # Pick up package if at truck and not carrying
                if drone.state.carrying_package is None:
                    drone.state.carrying_package = drone.action.target_customer

            if target is None:
                drone.state.p_vel = np.zeros(self.world.dim_p)
                continue

            # Calculate movement (drone uses L2 distance)
            direction = target - drone.state.p_pos
            distance = drone_distance(drone.state.p_pos, target)

            if distance > 1e-6:
                direction = direction / distance
                step_size = min(drone.max_speed * self.world.dt, distance)

                # Update position
                drone.state.p_pos += direction * step_size
                drone.state.p_vel = direction * (step_size / self.world.dt)

                # Consume battery
                battery_used = step_size * drone.battery_consumption_rate
                drone.state.battery -= battery_used
                drone.battery_used_this_step += battery_used

                # Check for battery depletion (crash)
                if drone.state.battery <= 0:
                    drone.state.battery = 0
                    drone.state.status = 'crashed'
                    drone.state.p_vel = np.zeros(self.world.dim_p)
            else:
                drone.state.p_vel = np.zeros(self.world.dim_p)

            # Check for recovery arrival
            if drone.action.return_to_truck:
                # Drone distance (L2) for recovery check
                dist_to_truck = drone_distance(
                    drone.state.p_pos, self.world.truck.state.p_pos
                )
                if dist_to_truck < self.world.recovery_threshold:
                    # Auto-recover when arriving at truck
                    drone.state.status = 'onboard'
                    drone.state.p_pos = self.world.truck.state.p_pos.copy()
                    drone.state.p_vel = np.zeros(self.world.dim_p)
                    drone.state.target_pos = None
                    drone.action.return_to_truck = False
                    # Partial recharge
                    drone.state.battery = min(1.0, drone.state.battery + 0.2)

    def _update_truck_state(self):
        """Update truck position along road path.

        Truck follows the actual road path from GraphHopper, ensuring
        the truck (and drones launched from it) are always on valid roads.
        """
        truck = self.world.truck

        # No target - stay in place
        if truck.state.target_node is None:
            truck.state.p_vel = np.zeros(self.world.dim_p)
            truck.state.road_path = None
            truck.state.path_index = 0
            return

        # Get target position from target_node
        target = self.world.route_nodes[truck.state.target_node]

        # Initialize road path if needed (new target or no path)
        if truck.state.road_path is None:
            self._init_truck_road_path(truck, target)

        # If still no path (GraphHopper failed), fall back to direct movement
        if truck.state.road_path is None or len(truck.state.road_path) == 0:
            self._update_truck_direct(truck, target)
            return

        # Move along the road path
        self._update_truck_along_path(truck, target)

    def _init_truck_road_path(self, truck, target):
        """Initialize road path from current position to target."""
        # Get road path as geographic coordinates
        geo_path = self.distance_calculator.get_truck_route_path(
            truck.state.p_pos, target,
            simplify=True,
            max_points=100  # Enough resolution for smooth movement
        )

        if geo_path is None or len(geo_path) < 2:
            # GraphHopper unavailable or failed
            truck.state.road_path = None
            truck.state.road_path_geo = None
            return

        # Store geographic path for p_geo synchronization
        truck.state.road_path_geo = geo_path

        # Convert geographic path to environment coordinates
        env_path = []
        for point in geo_path:
            # point is [lon, lat]
            env_pos = self.distance_calculator.geo_to_env(point[0], point[1])
            env_path.append(env_pos)

        truck.state.road_path = env_path
        truck.state.path_index = 0

        # Initialize p_geo to start of path
        truck.state.p_geo = np.array([geo_path[0][0], geo_path[0][1]])

    def _update_truck_along_path(self, truck, final_target):
        """Move truck along the cached road path."""
        path = truck.state.road_path
        remaining_move = truck.max_speed * self.world.dt
        total_distance_traveled = 0.0

        while remaining_move > 1e-6 and truck.state.path_index < len(path):
            # Current waypoint target
            waypoint = path[truck.state.path_index]
            direction = waypoint - truck.state.p_pos
            dist_to_waypoint = np.linalg.norm(direction)

            if dist_to_waypoint < 1e-6:
                # Already at this waypoint, move to next
                truck.state.path_index += 1
                continue

            if dist_to_waypoint <= remaining_move:
                # Can reach this waypoint, move to it and continue to next
                total_distance_traveled += dist_to_waypoint
                remaining_move -= dist_to_waypoint
                truck.state.p_pos = waypoint.copy()
                truck.state.path_index += 1
            else:
                # Move partially toward waypoint
                direction = direction / dist_to_waypoint
                truck.state.p_pos += direction * remaining_move
                total_distance_traveled += remaining_move
                truck.state.p_vel = direction * (remaining_move / self.world.dt)
                remaining_move = 0

        # Record distance traveled (already in env units along road)
        truck.distance_traveled_this_step = total_distance_traveled

        # Sync p_geo with current path_index
        if truck.state.road_path_geo is not None:
            idx = min(truck.state.path_index, len(truck.state.road_path_geo) - 1)
            geo_point = truck.state.road_path_geo[idx]
            truck.state.p_geo = np.array([geo_point[0], geo_point[1]])

        # Update onboard drones' positions
        for drone in self.world.drones:
            if drone.state.status == 'onboard':
                drone.state.p_pos = truck.state.p_pos.copy()

        # Check if arrived at final target
        dist_to_final = np.linalg.norm(final_target - truck.state.p_pos)
        if dist_to_final < 1e-4 or truck.state.path_index >= len(path):
            # Arrived at target node
            truck.state.p_pos = final_target.copy()
            truck.state.current_node = truck.state.target_node
            truck.state.target_node = None
            truck.state.road_path = None
            truck.state.road_path_geo = None
            truck.state.path_index = 0
            truck.state.p_vel = np.zeros(self.world.dim_p)
            # Update p_geo to final position
            geo = self.distance_calculator.env_to_geo(final_target)
            truck.state.p_geo = np.array([geo[0], geo[1]])

    def _update_truck_direct(self, truck, target):
        """Fallback: direct L2 movement when road path unavailable."""
        direction = target - truck.state.p_pos
        l2_distance = np.linalg.norm(direction)

        if l2_distance > 1e-6:
            direction = direction / l2_distance
            step_size = min(truck.max_speed * self.world.dt, l2_distance)

            truck.state.p_pos += direction * step_size
            truck.state.p_vel = direction * (step_size / self.world.dt)
            truck.distance_traveled_this_step = step_size

            # Sync p_geo using coordinate conversion
            geo = self.distance_calculator.env_to_geo(truck.state.p_pos)
            truck.state.p_geo = np.array([geo[0], geo[1]])

            # Update onboard drones
            for drone in self.world.drones:
                if drone.state.status == 'onboard':
                    drone.state.p_pos = truck.state.p_pos.copy()

            if np.linalg.norm(target - truck.state.p_pos) < 1e-6:
                truck.state.current_node = truck.state.target_node
                truck.state.target_node = None
        else:
            truck.state.current_node = truck.state.target_node
            truck.state.target_node = None
            truck.state.p_vel = np.zeros(self.world.dim_p)

    def _check_deliveries(self):
        """Check and process deliveries. Prevent multiple drones serving same customer."""
        for drone in self.world.drones:
            if drone.action.target_customer is None:
                continue
            if drone.state.carrying_package is None:
                continue

            customer_idx = drone.action.target_customer
            customer = self.world.customers[customer_idx]

            # Skip already served customers!
            if customer.state.served:
                continue

            # Drone distance (L2) for delivery check
            dist = drone_distance(drone.state.p_pos, customer.state.p_pos)
            if dist < self.world.delivery_threshold:
                # Complete delivery
                customer.state.served = True
                customer.state.arrival_step = self.current_step
                customer.just_served_this_step = True
                customer.color = np.array([0.25, 0.75, 0.25])  # Green when served

                # Drone delivered package
                drone.state.carrying_package = None
                drone.state.target_pos = None
                drone.action.target_customer = None

    def _compute_global_reward(self):
        """Compute global reward. Called ONCE per step."""
        if self.reward_callback is not None:
            return self.reward_callback(self.world)
        return 0.0

    def _is_terminal(self):
        """Check if episode should terminate."""
        if self.done_callback is not None:
            return self.done_callback(self.world)

        # Default: done when max steps reached
        return self.current_step >= self.world_length

    def _get_obs(self, agent):
        """Get observation for an agent, padded to max_obs_dim."""
        if self.observation_callback is not None:
            obs = self.observation_callback(agent, self.world)
            # Pad to max_obs_dim for compatibility with vectorized envs
            if len(obs) < self.max_obs_dim:
                padded_obs = np.zeros(self.max_obs_dim, dtype=np.float32)
                padded_obs[:len(obs)] = obs
                return padded_obs
            return obs
        return np.zeros(self.max_obs_dim, dtype=np.float32)

    def _get_share_obs(self):
        """Get shared observation (global state)."""
        if self.share_obs_callback is not None:
            return self.share_obs_callback(self.world)
        # Fallback: concatenate all observations
        all_obs = []
        for agent in self.agents:
            all_obs.extend(self._get_obs(agent))
        return np.array(all_obs, dtype=np.float32)

    def _get_info(self, agent):
        """Get info dict for an agent."""
        if self.info_callback is not None:
            return self.info_callback(agent, self.world)
        return {}

    def _get_available_actions(self, agent):
        """Get available actions mask for an agent."""
        if self.available_actions_callback is not None:
            return self.available_actions_callback(agent, self.world)
        # Default: all actions available
        return np.ones(self.action_space[self.agents.index(agent)].n)

    def close(self):
        """Clean up resources."""
        pass

    def render(self, mode='human'):
        """Render the environment (optional)."""
        # Basic text rendering
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Truck: pos={self.world.truck.state.p_pos}")
            for i, drone in enumerate(self.world.drones):
                print(f"Drone {i}: pos={drone.state.p_pos}, battery={drone.state.battery:.2f}, status={drone.state.status}")
            for i, customer in enumerate(self.world.customers):
                served_str = "SERVED" if customer.state.served else "waiting"
                print(f"Customer {i}: pos={customer.state.p_pos}, {served_str}")
        return None
