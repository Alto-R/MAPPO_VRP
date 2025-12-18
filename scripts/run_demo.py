"""
Simple demo script to run MAPPO on VRP environment and save timestep data.
Generates JSON data for visualization.

Usage:
    # Random actions (no model):
    python run_demo.py --scenario_name shenzhen_delivery --num_drones 2 --num_customers 10 --random

    # With trained model (auto-find latest):
    python run_demo.py --scenario_name shenzhen_delivery --use_graphhopper --depot_index 3 \
        --num_customers 10 --num_drones 2 --customer_radius_km 5.0 --road_radius_km 10.0

    # With specific model directory:
    python run_demo.py --model_dir "path/to/model/files" --num_drones 2 --num_customers 3
"""

import os
import sys
import json
import numpy as np
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mappo.envs.vrp.VRP_env import VRPEnv
from mappo.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


def create_policy_args():
    """Create minimal args for policy initialization."""
    class PolicyArgs:
        def __init__(self):
            # Network architecture
            self.hidden_size = 64
            self.layer_N = 1
            self.use_orthogonal = True
            self.use_ReLU = True
            self.use_feature_normalization = True
            self.gain = 0.01
            self.stacked_frames = 1

            # RNN settings
            self.use_naive_recurrent_policy = False
            self.use_recurrent_policy = False
            self.recurrent_N = 1
            self.data_chunk_length = 10

            # Policy settings
            self.use_policy_active_masks = True

            # Critic settings
            self.use_popart = False

            # Optimizer settings (not used for inference but required)
            self.lr = 5e-4
            self.critic_lr = 5e-4
            self.opti_eps = 1e-5
            self.weight_decay = 0
    return PolicyArgs()


def load_policies(args, env, device):
    """Load trained policies from model_dir."""
    policies = []
    num_agents = 1 + args.num_drones  # 1 truck + drones

    policy_args = create_policy_args()

    for agent_id in range(num_agents):
        obs_space = env.observation_space[agent_id]
        act_space = env.action_space[agent_id]

        # Create policy
        policy = Policy(policy_args, obs_space, obs_space, act_space, device=device)

        # Load actor weights
        actor_path = os.path.join(args.model_dir, f'actor_agent{agent_id}.pt')
        if os.path.exists(actor_path):
            policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
            print(f"Loaded actor for agent {agent_id}")
        else:
            raise FileNotFoundError(f"Actor weights not found: {actor_path}")

        policy.actor.eval()
        policies.append(policy)

    return policies


def run_demo(args):
    """Run a simple demo episode and save timestep data."""

    # Device setup
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = VRPEnv(args)

    # Load trained policies if model_dir provided
    policies = None
    if args.model_dir:
        print(f"Loading model from: {args.model_dir}")
        policies = load_policies(args, env, device)
    else:
        print("No model_dir provided, using random actions")

    # Storage for timestep data
    episode_data = {
        'config': {
            'num_drones': args.num_drones,
            'num_customers': args.num_customers,
            'num_route_nodes': args.num_route_nodes,
            'episode_length': args.episode_length,
        },
        'route_nodes': [],
        'timesteps': []
    }

    # Check if we have geographic data (shenzhen scenario)
    has_geo = hasattr(env.world, 'coord_converter') and env.world.coord_converter is not None
    if has_geo:
        geo_bounds = env.world.geo_bounds
        episode_data['geo_bounds'] = {
            'min_lon': float(geo_bounds[0]),
            'max_lon': float(geo_bounds[1]),
            'min_lat': float(geo_bounds[2]),
            'max_lat': float(geo_bounds[3])
        }
        episode_data['has_geo'] = True
        print(f"Geographic data available: lon=[{geo_bounds[0]:.5f}, {geo_bounds[1]:.5f}], lat=[{geo_bounds[2]:.5f}, {geo_bounds[3]:.5f}]")
    else:
        episode_data['has_geo'] = False

    # Helper function to convert env coords to geo coords
    def env_to_geo(pos):
        if has_geo:
            lon, lat = env.world.coord_converter.env_to_geo(pos)
            return {'lon': float(lon), 'lat': float(lat)}
        return None

    # Helper function to interpolate truck position along road path
    def get_truck_geo_on_road(truck_pos, current_node, target_node):
        """
        If truck is traveling between nodes and road path exists,
        interpolate position along the road path based on progress.
        """
        if not has_geo:
            return None

        # If not traveling or same node, just use direct conversion
        if current_node is None or target_node is None or current_node == target_node:
            return env_to_geo(truck_pos)

        path_key = f"{current_node}-{target_node}"
        if path_key not in road_paths or len(road_paths[path_key]) < 2:
            return env_to_geo(truck_pos)

        road_path = road_paths[path_key]

        # Calculate progress along the segment in env coordinates
        start_pos = env.world.route_nodes[current_node]
        end_pos = env.world.route_nodes[target_node]
        total_dist = np.linalg.norm(end_pos - start_pos)

        if total_dist < 1e-6:
            return env_to_geo(truck_pos)

        current_dist = np.linalg.norm(truck_pos - start_pos)
        progress = min(1.0, max(0.0, current_dist / total_dist))

        # Interpolate along road path
        path_idx = int(progress * (len(road_path) - 1))
        path_idx = min(path_idx, len(road_path) - 1)

        return {'lon': float(road_path[path_idx][0]), 'lat': float(road_path[path_idx][1])}

    # Store route nodes (fixed positions)
    for node in env.world.route_nodes:
        node_data = {
            'x': float(node[0]),
            'y': float(node[1])
        }
        if has_geo:
            geo = env_to_geo(node)
            node_data['lon'] = geo['lon']
            node_data['lat'] = geo['lat']
        episode_data['route_nodes'].append(node_data)

    # Pre-compute road paths between route nodes if GraphHopper is available
    road_paths = {}
    if has_geo and args.use_graphhopper:
        print("Pre-computing road paths between route nodes...")
        try:
            from mappo.envs.vrp.distance_utils import DistanceCalculator
            dist_calc = DistanceCalculator(
                use_graphhopper=True,
                graphhopper_url=args.graphhopper_url,
                geo_bounds=env.world.geo_bounds
            )

            num_nodes = len(env.world.route_nodes)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        path = dist_calc.get_truck_route_path(
                            env.world.route_nodes[i],
                            env.world.route_nodes[j],
                            simplify=True,
                            max_points=100
                        )
                        if path:
                            road_paths[f"{i}-{j}"] = path

            print(f"  Computed {len(road_paths)} road paths")
            episode_data['road_paths'] = road_paths
        except Exception as e:
            print(f"  Warning: Failed to compute road paths: {e}")
            episode_data['road_paths'] = {}
    else:
        episode_data['road_paths'] = {}

    # Reset environment
    obs_n = env.reset()

    # Store initial customer positions (these are randomized on reset)
    initial_customers = []
    for i, c in enumerate(env.world.customers):
        cust_data = {
            'id': i,
            'x': float(c.state.p_pos[0]),
            'y': float(c.state.p_pos[1]),
            'time_window_start': int(c.state.time_window_start),
            'time_window_end': int(c.state.time_window_end),
            'demand': float(c.state.demand)
        }
        if has_geo:
            geo = env_to_geo(c.state.p_pos)
            cust_data['lon'] = geo['lon']
            cust_data['lat'] = geo['lat']
            # Include POI name if available
            if hasattr(c, 'geo_info') and c.geo_info:
                cust_data['name'] = c.geo_info.get('name', f'Customer {i}')
        initial_customers.append(cust_data)
    episode_data['customers'] = initial_customers

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < args.episode_length:
        # Record current state
        truck_pos = env.world.truck.state.p_pos
        current_node = env.world.truck.state.current_node
        target_node = env.world.truck.state.target_node
        truck_data = {
            'x': float(truck_pos[0]),
            'y': float(truck_pos[1]),
            'vel_x': float(env.world.truck.state.p_vel[0]),
            'vel_y': float(env.world.truck.state.p_vel[1]),
            'current_node': int(current_node) if current_node is not None else None,
            'target_node': int(target_node) if target_node is not None else None
        }
        if has_geo:
            # Use road-interpolated position if available
            geo = get_truck_geo_on_road(truck_pos, current_node, target_node)
            truck_data['lon'] = geo['lon']
            truck_data['lat'] = geo['lat']

        timestep_data = {
            'step': step,
            'truck': truck_data,
            'drones': [],
            'customers_served': [],
            'actions': [],
            'reward': 0.0
        }

        # Record drone states
        for i, drone in enumerate(env.world.drones):
            carrying = drone.state.carrying_package
            drone_data = {
                'id': i,
                'x': float(drone.state.p_pos[0]),
                'y': float(drone.state.p_pos[1]),
                'battery': float(drone.state.battery),
                'status': drone.state.status,
                'carrying_package': int(carrying) if carrying is not None else None
            }
            if has_geo:
                geo = env_to_geo(drone.state.p_pos)
                drone_data['lon'] = geo['lon']
                drone_data['lat'] = geo['lat']
            timestep_data['drones'].append(drone_data)

        # Record which customers are served
        for i, c in enumerate(env.world.customers):
            if c.state.served:
                timestep_data['customers_served'].append(i)

        # Generate actions
        action_n = []
        for i, agent in enumerate(env.agents):
            # Get available actions mask
            avail = env._get_available_actions(agent)

            if policies is not None:
                # Use trained policy
                obs = torch.FloatTensor(obs_n[i]).unsqueeze(0).to(device)
                avail_tensor = torch.FloatTensor(avail).unsqueeze(0).to(device)
                rnn_states = torch.zeros(1, 1, policies[i].actor.hidden_size).to(device)
                masks = torch.ones(1, 1).to(device)

                with torch.no_grad():
                    action, _ = policies[i].act(obs, rnn_states, masks, avail_tensor, deterministic=True)
                action = action.cpu().numpy().flatten()[0]
            else:
                # Random action from available actions
                valid_actions = np.where(avail > 0)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0

            action_n.append(action)
            timestep_data['actions'].append(int(action))

        # Step environment
        obs_n, reward_n, done_n, info_n = env.step(action_n)
        done = done_n[0]
        reward = reward_n[0][0]
        total_reward += reward
        timestep_data['reward'] = float(reward)

        episode_data['timesteps'].append(timestep_data)
        step += 1

    # Add final state
    final_truck_pos = env.world.truck.state.p_pos
    final_current_node = env.world.truck.state.current_node
    final_target_node = env.world.truck.state.target_node
    final_truck_data = {
        'x': float(final_truck_pos[0]),
        'y': float(final_truck_pos[1]),
        'vel_x': float(env.world.truck.state.p_vel[0]),
        'vel_y': float(env.world.truck.state.p_vel[1]),
        'current_node': int(final_current_node) if final_current_node is not None else None,
        'target_node': int(final_target_node) if final_target_node is not None else None
    }
    if has_geo:
        geo = get_truck_geo_on_road(final_truck_pos, final_current_node, final_target_node)
        final_truck_data['lon'] = geo['lon']
        final_truck_data['lat'] = geo['lat']

    final_timestep = {
        'step': step,
        'truck': final_truck_data,
        'drones': [],
        'customers_served': [],
        'actions': [],
        'reward': 0.0
    }
    for i, drone in enumerate(env.world.drones):
        carrying = drone.state.carrying_package
        drone_data = {
            'id': i,
            'x': float(drone.state.p_pos[0]),
            'y': float(drone.state.p_pos[1]),
            'battery': float(drone.state.battery),
            'status': drone.state.status,
            'carrying_package': int(carrying) if carrying is not None else None
        }
        if has_geo:
            geo = env_to_geo(drone.state.p_pos)
            drone_data['lon'] = geo['lon']
            drone_data['lat'] = geo['lat']
        final_timestep['drones'].append(drone_data)
    for i, c in enumerate(env.world.customers):
        if c.state.served:
            final_timestep['customers_served'].append(i)
    episode_data['timesteps'].append(final_timestep)

    # Summary
    episode_data['summary'] = {
        'total_steps': step,
        'total_reward': float(total_reward),
        'customers_served': sum(1 for c in env.world.customers if c.state.served),
        'total_customers': len(env.world.customers)
    }

    # Save to JSON
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'visualization', 'vrp', 'public')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'episode_data.json')

    with open(output_path, 'w') as f:
        json.dump(episode_data, f, indent=2)

    print(f"Episode completed!")
    print(f"  Steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Customers served: {episode_data['summary']['customers_served']}/{episode_data['summary']['total_customers']}")
    print(f"  Data saved to: {output_path}")

    return episode_data


def find_latest_model_dir(scenario_name='shenzhen_delivery', algorithm='mappo', experiment='check'):
    """Find the latest model directory based on run number."""
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'VRP', scenario_name, algorithm, experiment
    )

    if not os.path.exists(base_dir):
        return None

    # Find all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run')]
    if not run_dirs:
        return None

    # Get latest run
    run_nums = [int(d.replace('run', '')) for d in run_dirs]
    latest_run = f'run{max(run_nums)}'
    model_dir = os.path.join(base_dir, latest_run, 'models')

    if os.path.exists(model_dir):
        return model_dir
    return None


def main():
    parser = argparse.ArgumentParser()

    # Scenario
    parser.add_argument('--scenario_name', type=str, default='shenzhen_delivery',
                        help="Scenario name (truck_drone_basic or shenzhen_delivery)")

    # VRP configuration
    parser.add_argument('--num_drones', type=int, default=2,
                        help="Number of drones")
    parser.add_argument('--num_customers', type=int, default=10,
                        help="Number of customers to serve")
    parser.add_argument('--num_route_nodes', type=int, default=5,
                        help="Number of route nodes for truck")
    parser.add_argument('--episode_length', type=int, default=200,
                        help="Max length for any episode")

    # Thresholds
    parser.add_argument('--delivery_threshold', type=float, default=0.05,
                        help="Distance threshold for delivery completion")
    parser.add_argument('--recovery_threshold', type=float, default=0.1,
                        help="Distance threshold for drone recovery")

    # Reward parameters
    parser.add_argument('--delivery_bonus', type=float, default=10.0,
                        help="Reward for successful delivery")
    parser.add_argument('--late_penalty', type=float, default=0.5,
                        help="Penalty per step for late delivery")
    parser.add_argument('--energy_cost', type=float, default=0.1,
                        help="Cost per unit of energy consumed")
    parser.add_argument('--completion_bonus', type=float, default=50.0,
                        help="Bonus for serving all customers")

    # GraphHopper distance calculation parameters
    parser.add_argument('--use_graphhopper', action='store_true', default=False,
                        help="Use GraphHopper for truck road distance calculation (default: False, use L2)")
    parser.add_argument('--graphhopper_url', type=str, default='http://localhost:8989',
                        help="GraphHopper service URL")
    parser.add_argument('--geo_bounds', type=str, default=None,
                        help="Geographic bounds for coordinate conversion: 'min_lon,max_lon,min_lat,max_lat'")

    # Shenzhen delivery scenario parameters
    parser.add_argument('--geojson_path', type=str,
                        default='data/poi_batch_1_final_[7480]_combined_5.0km.geojson',
                        help="Path to GeoJSON file containing POI data")
    parser.add_argument('--depot_index', type=int, default=3,
                        help="Index of express point to use as depot (default: 3 = 顺丰速运福田)")
    parser.add_argument('--customer_radius_km', type=float, default=5.0,
                        help="Radius in km for customer selection around depot")
    parser.add_argument('--road_radius_km', type=float, default=10.0,
                        help="Radius in km for road network generation around depot")

    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    # Model loading
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Directory containing trained model weights. If not specified, auto-finds latest.")
    parser.add_argument('--algorithm', type=str, default='mappo',
                        help="Algorithm name for auto-finding model (mappo/rmappo/ippo)")
    parser.add_argument('--experiment', type=str, default='check',
                        help="Experiment name for auto-finding model")
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="Use GPU if available")
    parser.add_argument('--random', action='store_true', default=False,
                        help="Use random actions instead of trained model")

    args = parser.parse_args()

    # Auto-find model directory if not specified
    if args.model_dir is None and not args.random:
        args.model_dir = find_latest_model_dir(args.scenario_name, args.algorithm, args.experiment)
        if args.model_dir:
            print(f"Auto-found model directory: {args.model_dir}")
        else:
            print("No trained model found. Use --random for random actions or specify --model_dir")
            return

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_demo(args)


if __name__ == '__main__':
    main()