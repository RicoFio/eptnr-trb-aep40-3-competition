import igraph as ig
import geopandas as gdp
from typing import Tuple, Union
import pickle
from .synthetic_datasets import SyntheticDatasets
from pathlib import Path


def load_dataset(which: SyntheticDatasets) -> Tuple[ig.Graph, Union[gdp.GeoDataFrame, dict]]:
    """
    Returns graph and [GeoDataFrame OR RewardDict] depending on the dataset:
        - ONE: Graph & GDF
        - TWO: Graph & GDF
        - THREE: Graph & GDF
        - FOUR: Graph & RewardDict
        - FIVE: Graph & RewardDict
    """
    BASE_PATH = Path(__file__).parent
    graph = ig.read(BASE_PATH / which.value / 'graph.picklez')

    if which in [SyntheticDatasets.ONE, SyntheticDatasets.TWO, SyntheticDatasets.THREE, SyntheticDatasets.SIX]:
        return graph, gdp.read_file(BASE_PATH / which.value / 'census_data.geojson')
    else:
        return graph, pickle.load(BASE_PATH / which.value / 'reward_dict.pkl')
