import igraph as ig
from .base_reward import BaseReward
import logging
import pandas as pd
from typing import Dict

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class CustomReward(BaseReward):

    def __init__(self, reward_dict: Dict, census_data: pd.DataFrame, com_threshold: float):
        self.reward_dict = reward_dict
        super().__init__(census_data, com_threshold)

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        pass

    def evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        """

        Args:
            com_threshold:
            g:
            census_data:
            groups:

        Returns:

        """
        transit_edges = g.es.select(type_ne='walk', active_eq=0)
        edge_list = [e.index for e in transit_edges]

        # Sort the list such that it represents one of the possible combinations
        edge_list.sort()

        # Convert edge list to tuple as a list is not hashable
        edge_tuples = tuple(edge_list)

        return self.reward_dict[edge_tuples]

    def _reward_scaling(self, reward: float) -> float:
        return reward
