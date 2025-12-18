"""
Core entities for Multi-echelon VRP environment.
Defines Truck, Drone, Customer, and World classes.
"""

import numpy as np


class EntityState:
    """Base state for all entities."""
    def __init__(self):
        self.p_pos = None  # 2D position [x, y]
        self.p_vel = None  # 2D velocity [vx, vy]


class TruckState(EntityState):
    """Truck-specific state."""
    def __init__(self):
        super().__init__()
        self.current_node = None  # Current node index in route graph
        self.target_node = None   # Target node index (persistent until reached or changed)
        # Road path following
        self.road_path = None     # List of [x, y] points along road (env coords)
        self.road_path_geo = None # List of [lon, lat] points along road (geo coords)
        self.path_index = 0       # Current index in road_path
        self.p_geo = None         # Current geographic position [lon, lat]


class DroneState(EntityState):
    """Drone-specific state."""
    def __init__(self):
        super().__init__()
        self.battery = None           # Remaining battery (0-1 normalized)
        self.carrying_package = None  # Customer index or None
        self.status = None            # 'onboard', 'flying', 'returning', 'crashed'
        self.target_pos = None        # Current target position [x, y]


class CustomerState:
    """Customer state."""
    def __init__(self):
        self.p_pos = None              # 2D position [x, y]
        self.served = False            # Whether customer has been served
        self.demand = None             # Package demand (weight/size, normalized)
        self.time_window_start = None  # Earliest service time (step)
        self.time_window_end = None    # Latest service time (step)
        self.arrival_step = None       # When delivery was made


class TruckAction:
    """Truck action container."""
    def __init__(self):
        self.move_target = None     # Target position to move to
        self.release_drone = None   # Drone index to release (or None)
        self.recover_drone = None   # Drone index to recover (or None)


class DroneAction:
    """Drone action container."""
    def __init__(self):
        self.target_customer = None  # Customer index to deliver to
        self.return_to_truck = False # Return to truck flag
        self.hover = False           # Stay in place


class Entity:
    """Base entity class."""
    def __init__(self):
        self.name = ''
        self.size = 0.05
        self.movable = True
        self.color = None


class Truck(Entity):
    """Truck (mother ship) agent."""
    def __init__(self):
        super().__init__()
        self.state = TruckState()
        self.action = TruckAction()
        self.max_speed = 1.0
        self.drone_capacity = 3    # Max drones it can carry
        self.color = np.array([0.25, 0.75, 0.25])  # Green

        # Per-step tracking flags
        self.distance_traveled_this_step = 0.0


class Drone(Entity):
    """Drone agent."""
    def __init__(self):
        super().__init__()
        self.state = DroneState()
        self.action = DroneAction()
        self.max_speed = 2.0       # Faster than truck
        self.max_battery = 1.0
        self.battery_consumption_rate = 0.01  # Per distance unit
        self.color = np.array([0.25, 0.25, 0.75])  # Blue

        # Per-step tracking flags
        self.battery_used_this_step = 0.0
        self.forced_return_this_step = False


class Customer(Entity):
    """Customer (delivery target)."""
    def __init__(self):
        super().__init__()
        self.state = CustomerState()
        self.movable = False
        self.color = np.array([0.75, 0.25, 0.25])  # Red when unserved

        # Per-step tracking flags
        self.just_served_this_step = False


class World:
    """Multi-echelon VRP World."""
    def __init__(self):
        self.truck = None
        self.drones = []
        self.customers = []
        self.route_nodes = []      # Predefined route nodes for truck

        self.dim_p = 2              # 2D position space
        self.dt = 0.1               # Time step for physics
        self.world_length = 200     # Max episode steps
        self.world_step = 0

        # Environment bounds
        self.bounds = np.array([-1.0, 1.0])  # x and y bounds

        # Thresholds
        self.delivery_threshold = 0.05    # Distance to complete delivery
        self.recovery_threshold = 0.1     # Distance for drone recovery

    @property
    def entities(self):
        """Returns all entities in the world."""
        entities = [self.truck] + self.drones + self.customers
        return [e for e in entities if e is not None]

    @property
    def policy_agents(self):
        """Returns all controllable agents (truck + drones) in fixed order."""
        agents = []
        if self.truck is not None:
            agents.append(self.truck)
        agents.extend(self.drones)
        return agents

    @property
    def num_agents(self):
        """Total number of controllable agents."""
        return len(self.policy_agents)

    def step(self):
        """Advance world state by one timestep."""
        self.world_step += 1
