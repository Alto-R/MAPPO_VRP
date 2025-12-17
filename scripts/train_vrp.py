#!/usr/bin/env python
"""
Training script for VRP environment with MAPPO.

Usage:
    python train_vrp.py --scenario_name truck_drone_basic --num_drones 2 --num_customers 3
"""

import sys
import os

# Add mappo parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import setproctitle
import numpy as np
from pathlib import Path
import torch

from mappo.config import get_config
from mappo.envs.vrp.VRP_env import VRPEnv
from mappo.envs.vrp.env_wrappers import VRPDummyVecEnv, VRPSubprocVecEnv


def make_train_env(all_args):
    """Create vectorized training environment."""
    def get_env_fn(rank):
        def init_env():
            env = VRPEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return VRPDummyVecEnv([get_env_fn(0)])
    else:
        return VRPSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    """Create vectorized evaluation environment."""
    def get_env_fn(rank):
        def init_env():
            env = VRPEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return VRPDummyVecEnv([get_env_fn(0)])
    else:
        return VRPSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """Parse VRP-specific arguments."""
    # Scenario
    parser.add_argument('--scenario_name', type=str, default='truck_drone_basic',
                        help="Scenario name (truck_drone_basic)")

    # VRP configuration
    parser.add_argument('--num_drones', type=int, default=5,
                        help="Number of drones")
    parser.add_argument('--num_customers', type=int, default=15,
                        help="Number of customers to serve")
    parser.add_argument('--num_route_nodes', type=int, default=5,
                        help="Number of route nodes for truck")

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

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Set environment name
    all_args.env_name = "VRP"

    # VRP requires separated policy due to different action spaces
    if all_args.share_policy:
        print("WARNING: VRP requires separated policy due to different action spaces.")
        print("Setting share_policy to False.")
        all_args.share_policy = False

    # Algorithm configuration
    if all_args.algorithm_name == "rmappo":
        print("Using rmappo, setting use_recurrent_policy to True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("Using mappo, setting use_recurrent_policy & use_naive_recurrent_policy to False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("Using ippo, setting use_centralized_V to False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(f"Algorithm {all_args.algorithm_name} not supported")

    # CUDA setup
    if all_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Run directory
    run_dir = Path(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../results"
    )) / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Create run subfolder
    exst_run_nums = [
        int(str(folder.name).split('run')[1])
        for folder in run_dir.iterdir()
        if str(folder.name).startswith('run')
    ] if run_dir.exists() else []

    if len(exst_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-VRP-{all_args.experiment_name}"
    )

    # Seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Create environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # Number of agents = 1 truck + num_drones
    num_agents = 1 + all_args.num_drones

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Use VRPRunner (separated policy mode)
    from mappo.runner.vrp_runner import VRPRunner as Runner

    runner = Runner(config)
    runner.run()

    # Cleanup
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # Save tensorboard summary and close
    runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
