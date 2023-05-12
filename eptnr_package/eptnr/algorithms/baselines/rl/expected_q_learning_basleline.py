from .abstract_q_learner_baseline import AbstractQLearner
import numpy as np
from typing import List
from tqdm import tqdm


class ExpectedQLearner(AbstractQLearner):

    def train(self, return_rewards_over_episodes: bool = True, verbose: bool = True) -> List[float]:
        if self.trained:
            raise RuntimeError("Cannot run training pipeline twice. Please create a new learner object")

        cum_rewards_over_episodes = []
        iterator = tqdm(range(self.episodes)) if verbose else range(self.episodes)

        for ep, _ in enumerate(iterator):
            ord_state = self.get_state_key(self.starting_state)
            rewards = 0
            while len(ord_state) != len(self.actions):

                epsilon = self.eps_schedule.get_current_eps()

                action = self.choose_action(ord_state, epsilon=epsilon)
                next_state, reward = self.step(ord_state, action)
                next_ord_state = self.get_state_key(next_state)
                rewards += reward
                # Q-Learning update
                self.q_values[ord_state][action] += self.alpha * (
                                                    reward + self.gamma * np.max(self.q_values[next_ord_state]) -
                                                    self.q_values[ord_state][action])
                ord_state = next_ord_state
            cum_rewards_over_episodes.append(rewards)

        self.trained = True

        if return_rewards_over_episodes:
            return cum_rewards_over_episodes
