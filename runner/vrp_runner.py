"""
VRP Runner for MAPPO training.
Inherits from base_runner and customizes for VRP.
"""

import time
import numpy as np
import torch

from mappo.runner.base_runner import Runner


def _t2n(x):
    """Convert tensor to numpy."""
    return x.detach().cpu().numpy()


class VRPRunner(Runner):
    """
    Runner for VRP environment.
    Customized for truck-drone delivery scenarios.
    """

    def __init__(self, config):
        super(VRPRunner, self).__init__(config)

        # Early stopping setup
        self.use_early_stopping = getattr(self.all_args, 'use_early_stopping', False)
        self.early_stop_patience = getattr(self.all_args, 'early_stop_patience', 50)
        self.early_stop_min_delta = getattr(self.all_args, 'early_stop_min_delta', 0.01)
        self.early_stop_window = getattr(self.all_args, 'early_stop_window', 10)

        # Early stopping state
        self.reward_history = []
        self.best_avg_reward = float('-inf')
        self.episodes_without_improvement = 0

    def run(self):
        """Main training loop."""
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # Learning rate decay
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # Rollout
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # Environment step
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # Insert data into buffer
                self.insert(data)

            # Compute returns and train
            self.compute()
            train_infos = self.train()

            # Post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # Save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(f"\n VRP Scenario {self.all_args.scenario_name} "
                      f"Algo {self.algorithm_name} Exp {self.experiment_name} "
                      f"Episode {episode}/{episodes}, "
                      f"Steps {total_num_steps}/{self.num_env_steps}, "
                      f"FPS {int(total_num_steps / (end - start))}\n")

                # VRP-specific metrics
                self._log_vrp_metrics(infos, train_infos)
                self.log_train(train_infos, total_num_steps)

            # Evaluation
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            # Early stopping check
            if self.use_early_stopping:
                should_stop, reason = self._check_early_stopping(train_infos)
                if should_stop:
                    print(f"\n{'='*50}")
                    print(f"Early stopping triggered at episode {episode}")
                    print(f"Reason: {reason}")
                    print(f"Best average reward: {self.best_avg_reward:.4f}")
                    print(f"{'='*50}\n")
                    self.save()  # Save final model
                    break

    def warmup(self):
        """Initialize buffer with first observation."""
        obs = self.envs.reset()

        # Do a dummy step to get share_obs from info
        # Use "stay" action (action 0) for all agents
        dummy_actions = []
        for i in range(self.n_rollout_threads):
            env_actions = []
            for agent_id in range(self.num_agents):
                # Create one-hot encoded action 0 (STAY/HOVER)
                action_env = np.zeros(self.envs.action_space[agent_id].n)
                action_env[0] = 1
                env_actions.append(action_env)
            dummy_actions.append(env_actions)

        # Step to get info with share_obs
        _, _, _, infos = self.envs.step(dummy_actions)
        share_obs = self._get_share_obs_from_infos(infos)

        # Reset to get fresh starting state
        obs = self.envs.reset()

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs_agent = np.array(list(obs[:, agent_id]))
            else:
                share_obs_agent = share_obs

            self.buffer[agent_id].share_obs[0] = share_obs_agent.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    def _get_share_obs_from_infos(self, infos):
        """
        Extract share_obs from info dicts returned by step.
        For VRP, all agents share the same global state.
        """
        share_obs_list = []
        for env_infos in infos:
            # Get share_obs from first agent's info (all agents have same share_obs)
            if env_infos and len(env_infos) > 0 and isinstance(env_infos[0], dict) and 'share_obs' in env_infos[0]:
                share_obs_list.append(env_infos[0]['share_obs'])
            else:
                # Fallback: create empty share_obs with expected dim
                share_obs_dim = self.envs.share_observation_space[0].shape[0]
                share_obs_list.append(np.zeros(share_obs_dim, dtype=np.float32))
        return np.array(share_obs_list)

    @torch.no_grad()
    def collect(self, step):
        """Collect data for a single step."""
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = \
                self.trainer[agent_id].policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step]
                )

            values.append(_t2n(value))
            action = _t2n(action)

            # Convert action to environment format (one-hot)
            if self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                action_env = None
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # Rearrange to [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        """Insert data into replay buffer."""
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        # Reset RNN states on done
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32
        )

        # Create masks
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # Get share_obs from info (world-level features)
        share_obs = self._get_share_obs_from_infos(infos)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs_agent = np.array(list(obs[:, agent_id]))
            else:
                share_obs_agent = share_obs

            self.buffer[agent_id].insert(
                share_obs_agent,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id]
            )

    def _log_vrp_metrics(self, infos, train_infos):
        """Log VRP-specific metrics."""
        customers_served = []
        total_customers = []

        for env_infos in infos:
            for agent_info in env_infos:
                if isinstance(agent_info, dict):
                    if 'customers_served' in agent_info:
                        customers_served.append(agent_info['customers_served'])
                    if 'total_customers' in agent_info:
                        total_customers.append(agent_info['total_customers'])

        if customers_served and total_customers:
            completion_rate = np.mean(customers_served) / np.mean(total_customers)
            print(f"  Customers served: {np.mean(customers_served):.2f}/{np.mean(total_customers):.2f} "
                  f"({completion_rate * 100:.1f}%)")

        for agent_id in range(self.num_agents):
            train_infos[agent_id].update({
                "average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length
            })
            print(f"  Agent {agent_id} average episode rewards: "
                  f"{train_infos[agent_id]['average_episode_rewards']:.2f}")

    def _check_early_stopping(self, train_infos):
        """
        Check if early stopping criteria are met.

        Returns:
            (should_stop, reason): Tuple of bool and string
        """
        # Calculate current episode reward (average across all agents)
        current_reward = np.mean([
            train_infos[agent_id].get('average_episode_rewards', 0)
            for agent_id in range(self.num_agents)
        ])

        # Add to history
        self.reward_history.append(current_reward)

        # Need at least window_size episodes to compute moving average
        if len(self.reward_history) < self.early_stop_window:
            return False, ""

        # Calculate moving average
        recent_rewards = self.reward_history[-self.early_stop_window:]
        current_avg_reward = np.mean(recent_rewards)

        # Check for improvement
        if current_avg_reward > self.best_avg_reward + self.early_stop_min_delta:
            self.best_avg_reward = current_avg_reward
            self.episodes_without_improvement = 0
            # Save best model
            self.save()
            print(f"  [Early Stop] New best avg reward: {self.best_avg_reward:.4f}")
        else:
            self.episodes_without_improvement += 1
            if self.episodes_without_improvement % 10 == 0:
                print(f"  [Early Stop] No improvement for {self.episodes_without_improvement} episodes "
                      f"(patience: {self.early_stop_patience})")

        # Check if patience exceeded
        if self.episodes_without_improvement >= self.early_stop_patience:
            return True, f"No improvement for {self.early_stop_patience} episodes"

        return False, ""

    @torch.no_grad()
    def eval(self, total_num_steps):
        """Evaluate the policy."""
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []

            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True
                )

                eval_action = eval_action.detach().cpu().numpy()

                # Convert action to environment format
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # Rearrange actions
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Step environment
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            # Reset RNN states on done
            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print(f"Eval average episode rewards of agent {agent_id}: {eval_average_episode_rewards:.2f}")

        self.log_train(eval_train_infos, total_num_steps)
