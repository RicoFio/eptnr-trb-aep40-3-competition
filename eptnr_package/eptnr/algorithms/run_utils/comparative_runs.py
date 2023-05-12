import os.path
import pickle
import random
import time

import pandas as pd
import torch

from eptnr.algorithms.baselines import (
    optimal_max_baseline,
    random_baseline,
    greedy_baseline,
    ga_max_baseline,
    ExpectedQLearner,
    MaxQLearner,
)
from eptnr.algorithms.deep_rl.deep_q_learning import (
    DeepQLearner,
    DeepMaxQLearner,
)
import igraph as ig
import numpy as np
import geopandas as gpd
from eptnr.rewards import BaseReward
import logging
from matplotlib import pyplot as plt
from eptnr.plotting.solution_plotting import plot_rewards_and_graphs
import torch
import random

from typing import Dict, Any
from pathlib import Path

from eptnr.datasets.dataset_loader import load_dataset
from eptnr.datasets.synthetic_datasets import SyntheticDatasets

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def comparative_runs(algorithms_and_kwargs: Dict[callable, Dict[str, Any]],
                     dataset_to_load: SyntheticDatasets, runs: int, base_seed: int = 2048,
                     save_output_to_path: Path = None):

    if os.path.exists(save_output_to_path):
        logger.warning(f"{save_output_to_path} exists, please make sure you're not overwriting existing results")
        time.sleep(5)

    g, census_data = load_dataset(dataset_to_load)
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    random_seeds = np.arange(0, runs) * base_seed
    results = {algo.__class_: [] for algo in algorithms_and_kwargs}

    all_solutions = []

    for rs in random_seeds:
        # Set random seed
        np.random.seed(rs)
        random.seed(rs)
        torch.manual_seed(rs)

        for algo, kwargs in algorithms_and_kwargs.items():
            result = algo(**kwargs)
            all_solutions.append(result)
            results[algo.__class_].append(max(result[0]))

        if runs == 1:
            plot_rewards_and_graphs(g, all_solutions, f"{' '.join(list(results.keys()))}",
                                    yticks=list(np.arange(0, 110, 10)))
            if save_output_to_path:
                plt.savefig(save_output_to_path)

            plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))

    for res in results:
        ax.plot(random_seeds, results[res], label=res, alpha=0.5, linestyle='-' if 'max' in res.lower() else '--')
        results[res] = results[res][:-1]

    ax.set_xticks(random_seeds[::5])
    ax.set_ylabel('Maximum reward')
    ax.set_xlabel('Random seed')
    ax.legend()
    fig.tight_layout()

    if save_output_to_path:
        plt.savefig(save_output_to_path)

    plt.show()

    latex = pd.DataFrame(results).describe().to_latex()
    if save_output_to_path:
        with open(save_output_to_path/'run_results.tex') as f:
            f.write(latex)
    logger.info(latex)

    if save_output_to_path:
        file_name = f'synth_ds_{dataset_to_load.name}_comparison_all_MAX_methods_over_random_seeds.svg'
        fig.savefig(save_output_to_path / file_name)
