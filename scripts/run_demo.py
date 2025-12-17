"""
Simple demo script to run MAPPO on VRP environment and save timestep data.
Generates JSON data for visualization.

Usage:
    # Random actions (no model):
    python run_demo.py --num_drones 2 --num_customers 3

    # With trained model:
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

    # Store route nodes (fixed positions)
    for node in env.world.route_nodes:
        episode_data['route_nodes'].append({
            'x': float(node[0]),
            'y': float(node[1])
        })

    # Reset environment
    obs_n = env.reset()

    # Store initial customer positions (these are randomized on reset)
    initial_customers = []
    for i, c in enumerate(env.world.customers):
        initial_customers.append({
            'id': i,
            'x': float(c.state.p_pos[0]),
            'y': float(c.state.p_pos[1]),
            'time_window_start': int(c.state.time_window_start),
            'time_window_end': int(c.state.time_window_end),
            'demand': float(c.state.demand)
        })
    episode_data['customers'] = initial_customers

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < args.episode_length:
        # Record current state
        timestep_data = {
            'step': step,
            'truck': {
                'x': float(env.world.truck.state.p_pos[0]),
                'y': float(env.world.truck.state.p_pos[1]),
                'vel_x': float(env.world.truck.state.p_vel[0]),
                'vel_y': float(env.world.truck.state.p_vel[1])
            },
            'drones': [],
            'customers_served': [],
            'actions': [],
            'reward': 0.0
        }

        # Record drone states
        for i, drone in enumerate(env.world.drones):
            carrying = drone.state.carrying_package
            timestep_data['drones'].append({
                'id': i,
                'x': float(drone.state.p_pos[0]),
                'y': float(drone.state.p_pos[1]),
                'battery': float(drone.state.battery),
                'status': drone.state.status,
                'carrying_package': int(carrying) if carrying is not None else None
            })

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
    final_timestep = {
        'step': step,
        'truck': {
            'x': float(env.world.truck.state.p_pos[0]),
            'y': float(env.world.truck.state.p_pos[1]),
            'vel_x': float(env.world.truck.state.p_vel[0]),
            'vel_y': float(env.world.truck.state.p_vel[1])
        },
        'drones': [],
        'customers_served': [],
        'actions': [],
        'reward': 0.0
    }
    for i, drone in enumerate(env.world.drones):
        carrying = drone.state.carrying_package
        final_timestep['drones'].append({
            'id': i,
            'x': float(drone.state.p_pos[0]),
            'y': float(drone.state.p_pos[1]),
            'battery': float(drone.state.battery),
            'status': drone.state.status,
            'carrying_package': int(carrying) if carrying is not None else None
        })
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


def find_latest_model_dir(scenario_name='truck_drone_basic', algorithm='mappo', experiment='check'):
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
    parser.add_argument('--scenario_name', type=str, default='truck_drone_basic')
    parser.add_argument('--num_drones', type=int, default=5)
    parser.add_argument('--num_customers', type=int, default=15)
    parser.add_argument('--num_route_nodes', type=int, default=5)
    parser.add_argument('--episode_length', type=int, default=10000)
    parser.add_argument('--delivery_threshold', type=float, default=0.05)
    parser.add_argument('--recovery_threshold', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

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