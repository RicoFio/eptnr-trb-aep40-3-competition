import igraph as ig
import numpy as np


def maximum_length_calculation(g: ig.Graph) -> float:
    g_prime: ig.Graph = g.copy()
    g_prime.es['tt'] = (-np.array(g_prime.es['tt'])).tolist()
    g_prime.get_shortest_paths()
    return
