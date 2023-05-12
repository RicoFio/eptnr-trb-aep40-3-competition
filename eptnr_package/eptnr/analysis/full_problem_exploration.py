import itertools as it
from typing import Tuple, List
import igraph as ig
import numpy as np
from ..rewards import BaseReward
from tqdm import tqdm
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def full_problem_exploration(g: ig.Graph, reward: BaseReward,
                             edge_types: List[str]) -> Tuple[List[List[List[int]]], List[List[float]]]:
    """

    Args:
        g:
        reward:
        edge_types:

    Returns:

    """
    num_transit_edges = len(g.es.select(type_ne='walk'))
    assert num_transit_edges > 0

    min_budget = 0
    max_budget = num_transit_edges - 1

    removable_edges = g.es.select(type_in=edge_types, active_eq=1)
    possible_combinations = [[[e.index for e in es] for es in it.combinations(removable_edges, budget)]
                             for budget in range(min_budget, max_budget+1)]
    rewards: List[List[float]] = [[-np.inf for c in g] for g in possible_combinations]
    configurations: List[List[int]] = [[None for c in g] for g in possible_combinations]

    logger.info(f"Possible states: {possible_combinations}")

    for i, candidates_k in enumerate(tqdm(possible_combinations)):
        for j, candidate in enumerate(candidates_k):
            g_prime = g.copy()
            g_prime.es[candidate]['active'] = 0
            configurations[i][j] = candidate
            r = reward.evaluate(g_prime)
            rewards[i][j] = r
            logger.info(f"For state {candidate} obtained rewards {rewards[i]}")

    return configurations, rewards
