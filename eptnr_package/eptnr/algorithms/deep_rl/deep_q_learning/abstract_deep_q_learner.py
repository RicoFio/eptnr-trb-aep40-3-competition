import datetime
import os.path

import igraph as ig
from typing import Tuple, List, Optional
import abc

import numpy as np
from .transition import Transition

import torch
from torch import nn
from ...baselines.rl.abstract_q_learner_baseline import AbstractQLearner
from ....rewards import BaseReward
from ...q_learning_utils.epsilon_schedule import EpsilonSchedule
from .model import DQN
from .replay_memory import ReplayMemory
import logging
import torch.optim as optim
import random
from pathlib import Path
from ....exceptions.q_learner_exceptions import ActionAlreadyTakenError

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class AbstractDeepQLearner(AbstractQLearner, abc.ABC):

    def __init__(self, base_graph: ig.Graph, reward: BaseReward, edge_types: List[str], episodes: int,
                 batch_size: int, replay_memory_size: int, target_network_update_step: int,
                 eps_start: float = 1.0, eps_end: float = 0.01, eps_decay: float = 200, static_eps_steps: int = 100,
                 step_size: float = 1, discount_factor: float = 1.0, save_model_every: int = 50,
                 save_model_path: Path = Path('./model_snapshots/')) -> None:
        super().__init__(base_graph, reward, edge_types, episodes, step_size, discount_factor)

        self.starting_state = torch.zeros(len(self.actions), dtype=torch.bool)
        self.wrong_action_reward: int = -100

        self.actions: torch.Tensor = torch.tensor(
            [e.index for e in self.base_graph.es.select(type_in=edge_types, active_eq=1)])

        # TODO convert this to entry parameter
        self.policy_net = DQN(len(self.actions)).to(device)
        self.target_net = DQN(len(self.actions)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        # self.target_net = DummyQTargetNet().evaluate()
        # self.target_net = DummyQTargetNet().evaluate()

        self.batch_size = batch_size
        self.target_update = target_network_update_step

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.eps_schedule = EpsilonSchedule(eps_start=eps_start, eps_end=eps_end,
                                            eps_decay=eps_decay, static_eps_steps=static_eps_steps)

        self.save_model_every = save_model_every

        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)

        self.save_model_path = save_model_path
        self.start_datetime = datetime.datetime.now().isoformat()

        self.policy_net_loss = []

        # TODO convert this to entry parameter (also learning rate)
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=1e-2, momentum=0.0)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-1)
        # TODO convert this to entry parameter
        self.criterion = nn.MSELoss()
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(self.replay_memory_size)

    def _get_q_value_dict(self):
        return {}

    def _increment_step(self):
        super(AbstractDeepQLearner, self)._increment_step()
        if self.curr_episode % self.save_model_every == 0 and self.curr_episode > 0:
            self.save_model(self.save_model_path.joinpath(f'model_{self.curr_episode}_{self.start_datetime}.pkl'))

    def _get_available_actions(self, state: torch.Tensor) -> List[int]:
        return [action_idx for action_idx, _ in enumerate(self.actions)
                if action_idx not in torch.where(state == 1)[0]]

    def _get_available_actions_boolean_tensor(self, available_actions: List[int]) -> torch.Tensor:
        available_actions_t = torch.zeros(len(self.actions), dtype=torch.bool)
        available_actions_t[available_actions] = 1
        return available_actions_t

    # choose an action based on epsilon greedy algorithm
    def choose_action(self, state: torch.Tensor, epsilon: float) -> torch.Tensor:
        available_actions = self._get_available_actions(state)
        available_actions_t = self._get_available_actions_boolean_tensor(available_actions)

        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state.float()).detach()  # .view(1,-1)

            q_values[~available_actions_t] = q_values.min() - 1
            max_idx = q_values.argmax().item()

            return torch.tensor([[max_idx]])
        else:
            return torch.tensor([[random.choice(available_actions)]], device=device, dtype=torch.long)

    def step(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Consider scaling the probabilities of not-allowed actions
        g_prime = self.base_graph.copy()
        action_idx = action.item()

        try:
            if state[action_idx].item() != 0:
                raise ActionAlreadyTakenError(
                    f"Cannot choose same action twice {action_idx} is already active in {state}"
                )

            # Create the next state
            next_state = state.detach().clone()
            next_state[action_idx] = 1

            # Select those edges from the actions which are removed
            edges = self.actions.masked_select(next_state).tolist()
            g_prime.es[edges]['active'] = 0

            # Evaluate graph
            reward = self.reward.evaluate(g_prime)
            reward = torch.tensor([[reward]], device=device, dtype=torch.float)

        except ActionAlreadyTakenError as e:
            logger.error(f"Reached a wrong state: {str(e)}")
            reward = torch.tensor([[self.wrong_action_reward]])
            next_state = self.starting_state
        self._increment_step()
        return next_state, reward

    @abc.abstractmethod
    def _compute_q_learning_targets(self, reward_batch: torch.tensor, next_state_values: torch.tensor) -> torch.tensor:
        raise NotImplementedError()

    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # # Compute a mask of non-final states and concatenate the batch elements
        # # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), device=device, dtype=torch.bool)
        # non_final_list = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(batch.next_state).view(len(batch.next_state), -1).float()
        state_batch = torch.cat(batch.state).view(len(batch.state), -1).float()
        available_actions_batch = torch.cat(batch.available_actions_next_state).view(
            len(batch.available_actions_next_state), -1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        target_q_values_next_state = self.target_net(non_final_next_states)
        target_q_values_next_state[~available_actions_batch] = target_q_values_next_state.min() - 1
        next_state_values = target_q_values_next_state.max(1)[0].detach()

        # TODO put this into an abstract function to be
        # TODO able to adapt this for MAX formulation
        # Compute the expected Q-learning targets for all actions
        expected_state_action_values = self._compute_q_learning_targets(reward_batch, next_state_values)

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.policy_net_loss.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clamping gradients
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    @abc.abstractmethod
    def train(self, return_cum_rewards_over_episodes: bool = True,
              return_max_rewards_over_episodes: bool = True,
              return_epsilon_over_episodes: bool = True,
              verbose: bool = True) -> Optional[Tuple[List[float], List[float], List[float]]]:
        raise NotImplementedError()

    def inference(self) -> Tuple[List[float], List[int]]:
        raise NotImplementedError()
