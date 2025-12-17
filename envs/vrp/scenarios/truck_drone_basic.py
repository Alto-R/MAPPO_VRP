"""
Basic Truck-Drone VRP Scenario for MVP.

1 Truck + 2-3 Drones + 3-5 Customers
Continuous 2D space, discrete time steps
"""

import numpy as np
from mappo.envs.vrp.core import World, Truck, Drone, Customer


class Scenario:
    """
    Basic truck-drone delivery scenario.
    """

    def __init__(self):
        # Reward parameters - minimize time objective
        self.time_penalty = 0.1           # Cost per time step (encourages speed)
        self.delivery_bonus = 5.0         # Reward per delivery
        self.completion_bonus = 100.0     # Big bonus for completing all deliveries
        self.incomplete_penalty = 20.0    # Penalty per unserved customer at end
        self.energy_cost = 0.01           # Small energy cost
        self.forced_return_penalty = 0.5  # Penalty for forced return

    def make_world(self, args):
        """
        Create and return the world.

        Args:
            args: Namespace with num_drones, num_customers, num_route_nodes, episode_length
        """
        world = World()
        world.world_length = getattr(args, 'episode_length', 200)

        # Get configuration
        num_drones = getattr(args, 'num_drones', 2)
        num_customers = getattr(args, 'num_customers', 3)
        num_route_nodes = getattr(args, 'num_route_nodes', 5)

        # Thresholds from args
        world.delivery_threshold = getattr(args, 'delivery_threshold', 0.05)
        world.recovery_threshold = getattr(args, 'recovery_threshold', 0.1)

        # Create truck
        world.truck = Truck()
        world.truck.name = 'truck_0'

        # Create drones
        world.drones = []
        for i in range(num_drones):
            drone = Drone()
            drone.name = f'drone_{i}'
            drone.state.status = 'onboard'
            world.drones.append(drone)

        # Create customers
        world.customers = []
        for i in range(num_customers):
            customer = Customer()
            customer.name = f'customer_{i}'
            world.customers.append(customer)

        # Generate route nodes for truck (fixed positions)
        world.route_nodes = self._generate_route_nodes(num_route_nodes)

        # Initialize world
        self.reset_world(world)

        return world

    def _generate_route_nodes(self, num_nodes):
        """
        Generate fixed route nodes for truck movement.
        Creates a circular pattern for MVP.
        """
        nodes = []
        for i in range(num_nodes):
            angle = 2 * np.pi * i / num_nodes
            radius = 0.6
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            nodes.append(np.array([x, y]))
        return nodes

    def reset_world(self, world):
        """Reset world to initial state."""
        # Reset truck
        world.truck.state.p_pos = np.array([0.0, 0.0])
        world.truck.state.p_vel = np.zeros(world.dim_p)
        world.truck.state.current_node = 0
        world.truck.state.target_node = None  # Persistent target node
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

        # Reset customers with random positions and time windows
        for i, customer in enumerate(world.customers):
            # Random positions within bounds
            customer.state.p_pos = np.random.uniform(-0.8, 0.8, world.dim_p)
            customer.state.served = False
            customer.state.demand = np.random.uniform(0.5, 1.0)

            # Time windows (in steps)
            tw_start = np.random.randint(0, world.world_length // 2)
            tw_duration = np.random.randint(world.world_length // 3, world.world_length)
            customer.state.time_window_start = tw_start
            customer.state.time_window_end = min(tw_start + tw_duration, world.world_length)
            customer.state.arrival_step = None
            customer.just_served_this_step = False

            # Update color based on served status
            customer.color = np.array([0.75, 0.25, 0.25])  # Red

        world.world_step = 0

    def observation(self, agent, world):
        """
        Generate observation for an agent.

        Returns:
            np.ndarray of observation features
        """
        obs = []

        # Self state (common for all agents)
        obs.extend(agent.state.p_pos)  # 2
        obs.extend(agent.state.p_vel)  # 2

        if isinstance(agent, Drone):
            # Drone-specific observation
            obs.append(agent.state.battery)  # 1
            obs.append(1.0 if agent.state.carrying_package is not None else 0.0)  # 1

            # Target position (or zeros if none)
            if agent.state.target_pos is not None:
                obs.extend(agent.state.target_pos)  # 2
            else:
                obs.extend([0.0, 0.0])  # 2

            # Onboard status
            obs.append(1.0 if agent.state.status == 'onboard' else 0.0)  # 1

            # Truck relative position
            rel_truck = world.truck.state.p_pos - agent.state.p_pos
            obs.extend(rel_truck)  # 2

            # Customer states (relative position)
            for customer in world.customers:
                rel_pos = customer.state.p_pos - agent.state.p_pos
                obs.extend(rel_pos)  # 2
                obs.append(1.0 if customer.state.served else 0.0)  # 1
                obs.append(self._time_window_remaining(customer, world))  # 1
                obs.append(customer.state.demand)  # 1

            # Other drones states
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
        """
        Generate shared observation (global state) for centralized critic.
        All agents receive the same share_obs.

        Returns:
            np.ndarray of global state features
        """
        share_obs = []

        # 1. Truck state (absolute coordinates)
        share_obs.extend(world.truck.state.p_pos)  # 2
        share_obs.extend(world.truck.state.p_vel)  # 2

        # 2. All drone states (absolute coordinates)
        for drone in world.drones:
            share_obs.extend(drone.state.p_pos)    # 2
            share_obs.extend(drone.state.p_vel)    # 2
            share_obs.append(drone.state.battery)  # 1
            share_obs.append(1.0 if drone.state.carrying_package is not None else 0.0)  # 1
            share_obs.append(self._encode_drone_status(drone.state.status))  # 1

        # 3. All customer states (absolute coordinates)
        for customer in world.customers:
            share_obs.extend(customer.state.p_pos)  # 2
            share_obs.append(1.0 if customer.state.served else 0.0)  # 1
            share_obs.append(self._time_window_remaining(customer, world))  # 1
            share_obs.append(customer.state.demand)  # 1

        # 4. Normalized time step
        share_obs.append(world.world_step / world.world_length)  # 1

        return np.array(share_obs, dtype=np.float32)

    def compute_share_obs_dim(self, world):
        """Compute the dimension of shared observation."""
        # 4 (truck) + num_drones * 7 + num_customers * 5 + 1 (time)
        return 4 + len(world.drones) * 7 + len(world.customers) * 5 + 1

    def get_available_actions(self, agent, world):
        """
        Return available actions mask (1=available, 0=unavailable).

        Args:
            agent: The agent (Truck or Drone)
            world: The world state

        Returns:
            np.ndarray mask
        """
        if isinstance(agent, Drone):
            # Drone action space: [HOVER, RETURN, DELIVER_0, DELIVER_1, ...]
            mask = np.ones(2 + len(world.customers))

            if agent.state.status == 'onboard':
                # On truck: can only HOVER (wait to be released)
                mask[1:] = 0  # Disable RETURN and all DELIVER
            elif agent.state.status == 'crashed':
                # Crashed: no actions available
                mask[:] = 0
                mask[0] = 1  # Can only hover (do nothing)
            else:
                # Flying
                # Disable already-served customers
                for i, c in enumerate(world.customers):
                    if c.state.served:
                        mask[2 + i] = 0

                # If carrying package, must deliver to target customer before returning
                if agent.state.carrying_package is not None:
                    # Disable RETURN action - must complete delivery first
                    mask[1] = 0
                    # Disable other customers - can only deliver to target
                    for i in range(len(world.customers)):
                        if i != agent.state.carrying_package:
                            mask[2 + i] = 0

        else:  # Truck
            # Truck action space: [STAY, MOVE_0..N, RELEASE_0..D, RECOVER_0..D]
            num_nodes = len(world.route_nodes)
            num_drones = len(world.drones)
            mask = np.ones(1 + num_nodes + 2 * num_drones)

            # RELEASE: only available for onboard drones
            for i, d in enumerate(world.drones):
                if d.state.status != 'onboard':
                    mask[1 + num_nodes + i] = 0

            # RECOVER: only available for nearby drones that are not onboard
            for i, d in enumerate(world.drones):
                dist = np.linalg.norm(d.state.p_pos - agent.state.p_pos)
                if dist > world.recovery_threshold or d.state.status == 'onboard':
                    mask[1 + num_nodes + num_drones + i] = 0

        return mask

    def is_terminal(self, world):
        """
        Check if episode should terminate.

        Returns:
            bool: True if episode is done
        """
        # Done if max steps reached
        if world.world_step >= world.world_length:
            return True

        # Done if all customers served
        if all(c.state.served for c in world.customers):
            return True

        # Done if all drones crashed
        if all(d.state.status == 'crashed' for d in world.drones):
            return True

        return False

    def compute_global_reward(self, world):
        """
        Compute global reward for cooperative scenario.
        Objective: minimize time to complete all deliveries.
        Called ONCE per step, all agents receive the same reward.

        Returns:
            float: The global reward
        """
        rew = 0.0

        # 1. Time penalty - encourages faster completion
        rew -= self.time_penalty

        # 2. Delivery rewards this step
        for c in world.customers:
            if c.just_served_this_step:
                rew += self.delivery_bonus

        # 3. Energy consumption (small)
        for d in world.drones:
            rew -= self.energy_cost * d.battery_used_this_step

        # 4. Forced return penalty
        for d in world.drones:
            if d.forced_return_this_step:
                rew -= self.forced_return_penalty

        # 5. Terminal reward
        if self.is_terminal(world):
            served = sum(1 for c in world.customers if c.state.served)
            total = len(world.customers)
            if served == total:
                # Big bonus for completing all deliveries
                # Earlier completion = fewer time penalties accumulated = higher total reward
                rew += self.completion_bonus
            else:
                # Penalty for incomplete deliveries
                rew -= self.incomplete_penalty * (total - served)

        return rew

    def info(self, agent, world):
        """Return info dict for an agent."""
        return {
            'customers_served': sum(1 for c in world.customers if c.state.served),
            'total_customers': len(world.customers),
            'time_step': world.world_step,
        }
