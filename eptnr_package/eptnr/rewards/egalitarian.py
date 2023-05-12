import igraph as ig
import statsmodels.api as sm
import numpy as np
from inequality.theil import TheilD
from .base_reward import BaseReward
import logging
import pandas as pd
from typing import List
from eptnr.constants.travel_metric import TravelMetric
from .utils.chebyshev_reward_computation import PartialRewardGenerator, chebyshev_reward_computation

logger = logging.getLogger(__name__)


class EgalitarianTheilReward(BaseReward):
    """

    """

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        metrics_dfs = self.retrieve_dfs(g)

        theil_inequality = {metric: None for metric in self.metrics_names}

        for metric, metric_df in metrics_dfs.items():
            X = metric_df.drop(columns='group').astype(float).to_numpy()
            Y = metric_df.group
            theil_t = TheilD(X, Y).T[0] if X.sum() > 0 else 0.0
            theil_inequality[metric] = theil_t

        return sum(theil_inequality.values())

    def _reward_scaling(self, reward: float) -> float:
        # Would be better if we could make this less random
        return -reward


class InverseTheilReward(BaseReward):
    """

    """

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        metrics_dfs = self.retrieve_dfs(g)

        theil_inequality = {metric: None for metric in self.metrics_names}

        for metric, metric_df in metrics_dfs.items():
            X = metric_df.drop(columns='group').astype(float).to_numpy()
            Y = metric_df.group.to_numpy()
            theil_t = TheilD(X, Y).T[0] if X.sum() > 0 else 0.0
            theil_inequality[metric] = theil_t

        return sum(theil_inequality.values())

    def _reward_scaling(self, reward: float) -> float:
        return np.log(len(self.census_data)) - reward


class EgalitarianTheilAndCostReward(BaseReward):

    def __init__(self, census_data: pd.DataFrame, com_threshold: float, total_graph_cost: float, monetary_budget: float,
                 groups: List[str] = None, metrics: List[TravelMetric] = None,
                 verbose: bool = False):
        super().__init__(census_data, com_threshold, groups, metrics, verbose)
        self.monetary_budget = monetary_budget
        self.theil_reward = EgalitarianTheilReward(census_data, com_threshold, groups, metrics, verbose, reward_scaling=False)
        self.theil_rg = PartialRewardGenerator(0, np.log(np.sum(self.census_data.n_inh)))
        if total_graph_cost - monetary_budget <= 0:
            raise ValueError("The monetary budget cannot be bigger than the cost of the total graph")
        self.cost_rg = PartialRewardGenerator(0, total_graph_cost)

    def _weight_calculation(self, min_rewards, max_rewards, ):
        pass

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        pass

    def _reward_scaling(self, reward: float) -> float:
        return reward

    def evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        """

        Args:
            g:

        Returns:

        """
        theil_reward = self.theil_reward.evaluate(g, *args, **kwargs)

        total_savings = sum(g.es.select(active=0)['cost'])

        partial_theil_reward = self.theil_rg.generate_reward(theil_reward)
        partial_cost_reward = self.cost_rg.generate_reward(abs(total_savings - self.monetary_budget))

        # We use the negative chebyshev as the reward is for a maximization problem but we want to
        # minimize both, the deviation from the budget and the inequality
        final_reward = -chebyshev_reward_computation(partial_theil_reward, partial_cost_reward)

        return final_reward
