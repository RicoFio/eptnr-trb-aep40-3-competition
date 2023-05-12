from typing import Tuple, List
import igraph as ig
from ...rewards import BaseReward


def greedy_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str]) -> Tuple[List[float], List[int]]:
    removable_edges = g.es.select(type_in=edge_types, active_eq=1)
    assert 0 < len(removable_edges)

    removed_edges = []
    rewards_per_removal = []

    g_prime = g.copy()

    for i in range(len(removable_edges)):
        removable_edges = g_prime.es.select(type_in=edge_types, active_eq=1)

        all_rewards = {}
        for edge in removable_edges:
            g_prime.es[edge.index]['active'] = 0
            r = reward.evaluate(g_prime)
            g_prime.es[edge.index]['active'] = 1
            all_rewards[r] = edge.index

        max_reward = max(all_rewards.keys())
        edge_to_remove = all_rewards[max_reward]
        removed_edges.append(edge_to_remove)
        g_prime.es[edge_to_remove]['active'] = 0
        rewards_per_removal.append(max_reward)

    return rewards_per_removal, removed_edges


def greedy_max_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str]) -> Tuple[List[float], List[int]]:
    rewards_per_removal, removed_edges = greedy_baseline(g, reward, edge_types)
    max_reward_index = rewards_per_removal.index(max(rewards_per_removal)) + 1

    return rewards_per_removal[:max_reward_index], removed_edges[:max_reward_index]
