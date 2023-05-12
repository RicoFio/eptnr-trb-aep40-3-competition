import igraph as ig
import numpy as np
from typing import (
    List,
    Tuple,
    Dict
)
from .synthetic_speeds import SyntheticTravelSpeeds
from ...constants.igraph_edge_types import IGraphEdgeTypes
from ...constants.igraph_colors import IGraphColors
from ...constants.igraph_vertex_types import IGraphVertexTypes
from enum import Enum
from ...constants.gtfs_network_costs_per_distance_unit import GTFSNetworkCostsPerDistanceUnit
from .eptnr_vertex import EPTNRVertex
import itertools as it


def set_eptnr_vertices(graph: ig.Graph, vertices: List[EPTNRVertex]):
    for i, vertex in enumerate(vertices):
        vertex.vertex_index = i

        V_O_attr = {
            'name': vertex.name,
            'x': vertex.x,
            'y': vertex.y,
            'color': vertex.color.value,
            'type': vertex.type.value,
        }
        graph.add_vertices(1, V_O_attr)


def compute_dist_from_es(g: ig.Graph, es: List[Tuple[str, str]], round_to_decimals: int = 2):
    get_coords = lambda vt: (g.vs.find(vt[0])['x'], g.vs.find(vt[0])['y'], g.vs.find(vt[1])['x'], g.vs.find(vt[1])['y'])
    positions = [get_coords(vertex_tuple) for vertex_tuple in es]
    pos_arr = np.array(positions)
    distances = np.sqrt((pos_arr[:, [0, 2]].max(axis=1) - pos_arr[:, [0, 2]].min(axis=1)) ** 2 +
                        (pos_arr[:, [1, 3]].max(axis=1) - pos_arr[:, [1, 3]].min(axis=1)) ** 2)
    return np.round(distances, decimals=round_to_decimals)


def graph_edge_entry(graph: ig.Graph, edges: List[Tuple[EPTNRVertex, EPTNRVertex]], names: List[str],
                     speed: SyntheticTravelSpeeds, edge_type: IGraphEdgeTypes,
                     cost: Enum = GTFSNetworkCostsPerDistanceUnit, color: IGraphColors = IGraphColors.BLACK,
                     round_to_decimals: int = 2, fixed_travel_times: list[float] = None,
                     fixed_costs: list[float] = None) -> None:
    """
    Commodity function to add edges to `graph`
    """
    edges_with_v_names = [(vs[0].name, vs[1].name) for vs in edges]
    distances= compute_dist_from_es(graph, edges_with_v_names)
    travel_times = fixed_travel_times or np.round((distances * 1 / speed.value) * 60, decimals=round_to_decimals)
    cost = fixed_costs or distances * cost[edge_type.name].value
    edges_attrs = {
        'name': names or None,
        'distance': distances,
        'tt': travel_times,
        'weight': travel_times,
        'cost': cost,
        'color': color.value,
        'type': edge_type.value,
    }
    graph.add_edges(edges_with_v_names, edges_attrs)


def graph_walking_edges_generation(graph: ig.Graph, vertices: List[EPTNRVertex]):
    rc_vertices = [rcv for rcv in filter(lambda x: x.type == IGraphVertexTypes.RC_NODE, vertices)]
    pt_vertices = [ptv for ptv in filter(lambda x: x.type == IGraphVertexTypes.PT_NODE, vertices)]
    poi_vertices = [poiv for poiv in filter(lambda x: x.type == IGraphVertexTypes.POI_NODE, vertices)]

    E_WALK = list(it.product(rc_vertices, pt_vertices)) + \
             list(it.product(pt_vertices, poi_vertices)) + \
             list(it.product(rc_vertices, poi_vertices))

    graph_edge_entry(
        graph=graph,
        edges=E_WALK,
        names=[],
        speed=SyntheticTravelSpeeds.WALKING_SPEED,
        edge_type=IGraphEdgeTypes.WALK,
        color=IGraphColors.GRAY,
        round_to_decimals=2,
    )


def add_cost_per_unit(graph: ig.Graph, cost_per_unit_dict: Dict[IGraphEdgeTypes, float]):

    cost_gen = [edge['distance'] * cost_per_unit_dict.get(IGraphEdgeTypes(edge['type']), 0) for edge in graph.es]
    graph.es.set_attribute_values('cost', cost_gen)
