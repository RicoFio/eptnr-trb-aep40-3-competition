import numpy as np
import logging
from tqdm import tqdm
from typing import (
    Tuple,
    List,
    Dict,
    Any,
)
import random
import torch

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def multi_run(algo: callable, n_runs: int, initial_seed: int = 2048,
              optimal_edges: List[List[int]] = None, verbose: bool = True) -> List[Tuple[List[float], List[int]]]:
    """
    Commodity function to run an algorithm multiple times with different random seeds.
    Args:
        algo: Learning algorithm returning Tuple[List[float], List[int]] which are rewards over removal
              and the removed edges.
        n_runs: Number of runs with different random seeds
        initial_seed:  [Optional] Initial random seed used as a basis to generate all subsequent random seeds.
                                  Default is 2048
        optimal_edges: [Optional] The list of all optimal edge configurations for the graph passed to the algo
        verbose: [Optional] Extensive logging; True by default
    Returns:
        A list of the identified solutions, i.e. list of rewards over removals and the corresponding edge list
    """
    random_seeds = np.arange(0, n_runs)*initial_seed
    successful = 0

    logger.info("Proceeding with multi-run experiments")

    multi_run_solutions = []

    for r in tqdm(range(n_runs)):
        # Set all random seeds of the used packages to be sure we have a replicable setting
        np.random.seed(random_seeds[r])
        random.seed(random_seeds[r])
        torch.manual_seed(random_seeds[r])

        rewards, edges = algo()
        multi_run_solutions.append([rewards, edges])

        if verbose:
            logger.info(f"Generated rewards: {rewards}\n"
                        f"Generated edges: {edges}")

            if optimal_edges:
                edge_diffs = [set(edges).symmetric_difference(opt) for opt in optimal_edges]
                if any([ed == set() for ed in edge_diffs]):
                    logger.info("This solution is optimal")
                    successful += 1
                else:
                    for diff in edge_diffs:
                        logger.info(f"\nThis solution is not optimal:\n"
                                    f"Solution {r}:\t{edges}\n"
                                    f"Optimal solutions:\t{optimal_edges}\n"
                                    f"Difference:\t{diff}")
    if optimal_edges and verbose:
        logger.info(f"Finds optimum in {successful} out of {n_runs}; {successful*100/n_runs}%")

    return multi_run_solutions
