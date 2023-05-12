import abc
import igraph as ig
import pandas as pd
from typing import List, Dict
from ..constants.travel_metric import TravelMetric
from .utils.graph_computation_utils import get_tt_hops_com_dfs
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class BaseReward(abc.ABC):

    def __init__(self, census_data: pd.DataFrame, com_threshold: float, groups: List[str] = None,
                 metrics: List[TravelMetric] = None, verbose: bool = False, reward_scaling=False) -> None:

        self.census_data = census_data
        self.com_threshold = com_threshold

        self.metrics = metrics or [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]
        self.metrics_names = [t.value for t in self.metrics]
        self.verbose = verbose
        self.reward_scaling = reward_scaling

        self.groups = groups

    def retrieve_dfs(self, g: ig.Graph) -> Dict[str, pd.DataFrame]:
        g_prime = g.subgraph_edges(g.es.select(active_eq=1), delete_vertices=False)
        tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g_prime, self.census_data, self.com_threshold)

        metrics_values = {
            TravelMetric.TT.value: tt_samples,
            TravelMetric.HOPS.value: hops_samples,
            TravelMetric.COM.value: com_samples
        }

        metrics_dfs = {metrics_name: metrics_values[metrics_name] for metrics_name in self.metrics_names}

        first_available_metric = list(metrics_dfs.keys())[0]
        self.groups = list(metrics_dfs[first_available_metric].group.unique()) if not self.groups else self.groups
        assert isinstance(self.groups, list)

        return metrics_dfs

    @abc.abstractmethod
    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def _reward_scaling(self, reward: float) -> float:
        raise NotImplementedError()

    def evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        g_prime = g.subgraph_edges(g.es.select(active_eq=1), delete_vertices=False)
        if self.verbose:
            logger.info("Received graph with:\n"
                        f"\tn_edges={len(g.es)}"
                        f"\tn_vertices={len(g.vs)}\n"
                        f"Created subgraph:\n"
                        f"\tn_edges={len(g_prime.es)}\n"
                        f"\tn_vertices={len(g_prime.vs)}")
        calculated_reward = self._evaluate(g_prime, *args, **kwargs)
        if self.reward_scaling:
            scaled_reward = self._reward_scaling(calculated_reward)
        else:
            scaled_reward = calculated_reward
        if self.verbose:
            logger.info(f"Resulting rewards:\n"
                        f"\t{calculated_reward=}\n"
                        f"\t{scaled_reward=}")

        return scaled_reward
