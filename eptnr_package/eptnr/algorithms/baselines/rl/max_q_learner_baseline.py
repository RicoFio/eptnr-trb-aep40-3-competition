from .abstract_q_learner_baseline import AbstractQLearner
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class MaxQLearner(AbstractQLearner):

    def train(self, return_rewards_over_episodes: bool = False, verbose: bool = True) -> List[float]:
        if self.trained:
            raise RuntimeError("Cannot run training pipeline twice. Please create a new learner object")

        max_rewards_over_episodes = []

        iterator = tqdm(range(self.episodes)) if verbose else range(self.episodes)
        for i in iterator:
            ord_state = self.get_state_key(self.starting_state)
            max_reward = -np.inf

            while len(ord_state) != len(self.actions):
                epsilon = self.eps_schedule.get_current_eps()
                action = self.choose_action(ord_state, epsilon=epsilon)
                next_state, reward = self.step(ord_state, action)
                next_ord_state = self.get_state_key(next_state)

                max_reward = np.max([max_reward, reward])

                # MaxQ-Learning update according to https://arxiv.org/abs/2010.03744
                self.q_values[ord_state][action] += self.alpha * (
                                                    np.max([reward, self.gamma * np.max(self.q_values[next_ord_state])]) -
                                                    self.q_values[ord_state][action])
                ord_state = next_ord_state
            max_rewards_over_episodes.append(max_reward)

        self.trained = True

        if return_rewards_over_episodes:
            return max_rewards_over_episodes

    def inference(self) -> Tuple[List[float], List[int]]:
        rewards_per_removal, final_state = super(MaxQLearner, self).inference()

        # Crop until max
        max_idx = np.argmax(rewards_per_removal) + 1

        return rewards_per_removal[:max_idx], final_state[:max_idx]
