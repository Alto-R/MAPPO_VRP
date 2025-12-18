"""
VRP Environment Factory.
Creates MultiAgentVRPEnv instances with appropriate scenario.
"""

from mappo.envs.vrp.environment import MultiAgentVRPEnv
from mappo.envs.vrp.scenarios import load


def VRPEnv(args):
    """
    Create a MultiAgentVRPEnv with the specified scenario.

    Args:
        args: Namespace containing:
            - scenario_name: Name of scenario file (without .py extension)
            - num_drones: Number of drone agents
            - num_customers: Number of customers
            - num_route_nodes: Number of route nodes for truck
            - episode_length: Max episode steps
            - delivery_threshold: Distance threshold for delivery
            - recovery_threshold: Distance threshold for drone recovery
            - use_graphhopper: Whether to use GraphHopper for truck distance (default: True)
            - graphhopper_url: GraphHopper service URL (default: http://localhost:8989)
            - geo_bounds: Geographic bounds for coordinate conversion (min_lon, max_lon, min_lat, max_lat)

    Returns:
        MultiAgentVRPEnv instance
    """
    # Load scenario
    scenario_name = getattr(args, 'scenario_name', 'truck_drone_basic')
    scenario = load(scenario_name + ".py").Scenario()

    # Create world
    world = scenario.make_world(args)

    # GraphHopper settings
    use_graphhopper = getattr(args, 'use_graphhopper', False)
    graphhopper_url = getattr(args, 'graphhopper_url', 'http://localhost:8989')

    # Parse geo_bounds from string format "min_lon,max_lon,min_lat,max_lat"
    geo_bounds_str = getattr(args, 'geo_bounds', None)
    geo_bounds = None
    if geo_bounds_str:
        try:
            bounds = [float(x.strip()) for x in geo_bounds_str.split(',')]
            if len(bounds) == 4:
                geo_bounds = tuple(bounds)
        except ValueError:
            print(f"[Warning] Invalid geo_bounds format: {geo_bounds_str}, using default")

    # Create environment with callbacks
    env = MultiAgentVRPEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.compute_global_reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info,
        done_callback=scenario.is_terminal,
        available_actions_callback=scenario.get_available_actions,
        share_obs_callback=scenario.get_share_obs,
        discrete_action=True,
        use_graphhopper=use_graphhopper,
        graphhopper_url=graphhopper_url,
        geo_bounds=geo_bounds
    )

    return env
