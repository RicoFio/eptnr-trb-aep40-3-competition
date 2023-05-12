from unittest import TestCase

import pandas as pd
import igraph as ig
from typing import Tuple


class RewardBaseTestCase(TestCase):

    def __init__(self):
        super().__init__()

    def get_completely_equal_overall_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_equal_tt_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_equal_hops_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_equal_com_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_completely_unequal_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_unequal_tt_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_unequal_hops_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_unequal_com_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_group_dominance_overall_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_group_dominance_tt_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_group_dominance_hops_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass

    def get_group_dominance_com_example(self) -> Tuple[ig.Graph, pd.DataFrame]:
        pass
