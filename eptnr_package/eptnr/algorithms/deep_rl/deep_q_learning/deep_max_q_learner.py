import math
from typing import Optional, List, Tuple

import numpy as np
import torch
from .abstract_deep_q_learner import AbstractDeepQLearner
import logging
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepMaxQLearner(AbstractDeepQLearner):

    def _compute_q_learning_targets(self, reward_batch: torch.tensor, next_state_values: torch.tensor) -> torch.tensor:
        return torch.max(reward_batch, (self.gamma * next_state_values))

    def train(self, return_cum_rewards_over_episodes: bool = False,
              return_max_rewards_over_episodes: bool = False,
              return_epsilon_over_episodes: bool = False,
              verbose: bool = False) -> Optional[Tuple[List[float], List[float], List[float]]]:

        cum_rewards_over_episodes = []
        max_rewards_over_episodes = []
        eps_values = []

        iterator_range = range(self.curr_episode, self.episodes)
        episode_iterator = tqdm(iterator_range) if verbose else iterator_range

        # for each episode
        for i_episode, ep in enumerate(episode_iterator):
            state = self.starting_state
            cum_reward = 0
            max_reward = -math.inf
            # while the state is not the terminal state
            while state[state == 1].size()[0] < len(self.actions):

                # Choose action with epsilon-greedy strategy
                epsilon = self.eps_schedule.get_current_eps()
                action_ = self.choose_action(state, epsilon)
                next_state, reward = self.step(state, action_)

                cum_reward += reward.item()
                max_reward = max([reward.item(), max_reward])

                # Store this transitions as an experience in the replay buffer
                # if len(self.memory) < self.replay_memory_size:
                available_actions_next_state = self._get_available_actions(next_state)
                available_actions_next_state_t = self._get_available_actions_boolean_tensor(available_actions_next_state)
                # 'state', 'action', 'next_state', 'available_actions_next_state', 'reward'
                self.memory.push(state, action_, next_state, available_actions_next_state_t, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            self.curr_episode = i_episode
            curr_eps_value = self.eps_schedule.get_current_eps()
            eps_values.append(curr_eps_value)
            cum_rewards_over_episodes.append(cum_reward)
            max_rewards_over_episodes.append(max_reward)

            if verbose:
                logger.info(f'#######')
                logger.info(f'Currently in episode {self.curr_episode}')
                logger.info(f'\tEpsilon value: {curr_eps_value}')
                logger.info(f'\tCumulative reward: {cum_reward}')
                logger.info(f'\tMaximum reward: {max_reward}')
                logger.info(f'\tPolicy network loss: {"" if not self.policy_net_loss else self.policy_net_loss[-1]}')
                logger.info(f'#######\n')

        self.trained = True

        output = []
        if return_cum_rewards_over_episodes:
            output.append(cum_rewards_over_episodes)

        if return_max_rewards_over_episodes:
            output.append(max_rewards_over_episodes)

        if return_epsilon_over_episodes:
            output.append(eps_values)

        return tuple(output)

    def inference(self) -> Tuple[List[float], List[int]]:
        if not self.trained:
            raise RuntimeError("Please run the training before inference")
        state = self.starting_state
        taken_actions = []
        rewards_per_removal = []

        for i in range(len(self.actions)):
            action_idx = self.choose_action(state, 0)
            taken_actions.append(action_idx.item())
            state, reward = self.step(state, action_idx)
            rewards_per_removal.append(reward.item())

        # A state is nothing else than an indicator as of whether an edge
        # is removed or not, i.e. whether an action was enacted or not.
        # Hence, we use the state to take out the actions, i.e. edge indexes
        # which represent the final state
        final_state = self.actions[taken_actions].numpy()

        # Crop until max
        max_idx = np.argmax(rewards_per_removal) + 1

        return rewards_per_removal[:max_idx], final_state[:max_idx]
