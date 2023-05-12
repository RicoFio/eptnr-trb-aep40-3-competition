from datetime import datetime

import geopandas as gpd
from typing import List
import igraph as ig
from enum import Enum

import networkx as nx
from pathlib import Path
from ..constants.osm_network_types import OSMNetworkTypes
from ..constants.gtfs_network_types import GTFSNetworkTypes
from ..constants.gtfs_network_costs_per_distance_unit import GTFSNetworkCostsPerDistanceUnit
from .gtfs_graph_generator import GTFSGraphGenerator
from .osm_graph_generation import OSMGraphGenerator
from ..constants.travel_speed import MetricTravelSpeed
from .utils.graph_expansion import add_points_to_graph, add_edges_to_graph
import logging
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ProblemGraphGenerator:
    # TODO change dat into ENUM
    # TODO change modalities into ENUM
    # TODO change distances_computation_mode into ENUM
    def __init__(self, city: str, gtfs_zip_file_path: Path, out_dir_path: Path,
                 day: str, time_from: str, time_to: str, agencies: List[str],
                 poi_gdf: gpd.GeoDataFrame, census_gdf: gpd.GeoDataFrame, modalities: List[str] = None,
                 distances_computation_mode: str = 'osmnx',
                 costs: Enum = GTFSNetworkCostsPerDistanceUnit) -> None:
        """

        Args:
            city:
            gtfs_zip_file_path:
            out_dir_path:
            day:
            time_from:
            time_to:
            poi_gdf:
            census_gdf:
            modalities: A list of modalities (e.g. ['tram', 'metro', 'bus']) to filter for. If None all modalities
                        are regarded.
            costs: The cost per distance unit (e.g. per meter) for each GTFS type (or the modality selected)
        """
        self.city = city
        self.gtfs_graph_generator = GTFSGraphGenerator(city=city, gtfs_zip_file_path=gtfs_zip_file_path,
                                                       out_dir_path=out_dir_path, day=day,
                                                       time_from=time_from, time_to=time_to, agencies=agencies,
                                                       contract_vertices=True, modalities=modalities, costs=costs)
        self.osm_graph_generator = OSMGraphGenerator(city=city, network_type=OSMNetworkTypes.WALK,
                                                     graph_out_path=out_dir_path)
        self.out_dir_path = out_dir_path
        self.poi_gdf = poi_gdf
        self.census_gdf = census_gdf
        self.distances_computation_mode = distances_computation_mode

    def generate_problem_graph(self) -> Path:
        """

        Returns:

        """

        # Generate GTFS Graph
        start_time = datetime.now()
        logger.debug(f"Starting GTFS graph generation on {start_time}")
        gtfs_graph_file_path = self.gtfs_graph_generator.generate_and_store_graph()
        logger.debug(f"Created GTFS graph and stored in {gtfs_graph_file_path}")

        # Load GTFS Graph
        logger.debug("Loading GTFS graph")
        pt_graph = ig.read(gtfs_graph_file_path)

        # Generate OSM Graph
        logger.debug("Starting OSM graph generation")
        osm_graph_file_path = self.osm_graph_generator.generate_and_store_graph()
        logger.debug(f"Created OSM Graph and stored in {osm_graph_file_path}")
        # Load OSM Graph
        logger.debug("Loading OSM graph")
        osm_graph = nx.read_gpickle(osm_graph_file_path)

        # Build new graph starting from the GTFS graph
        logger.debug("###\nStarting problem graph generation")
        g: ig.Graph = pt_graph.copy()
        # Set all existing vertices to be of type public transport
        g.vs.set_attribute_values(attrname='type', values='pt_node')

        # Add all residential centroids as vertices
        logger.debug("Adding residential centroid vertices to graph")
        rc_names = self.census_gdf.name.to_list()
        rc_xs = self.census_gdf.geometry.x.to_numpy()
        rc_ys = self.census_gdf.geometry.y.to_numpy()

        if not len(set(rc_names)) == len(rc_names):
            raise ValueError("Names of residential centroids in the GeoDataFrames have to be unique")

        # Names have to be integers!
        add_points_to_graph(g=g, names=rc_names, xs=rc_xs, ys=rc_ys,
                            v_type='rc_node', color='RED', ref_name=rc_names)

        # Add all POIs as vertices
        logger.debug("Adding POI vertices to graph")
        poi_names = self.poi_gdf.name.to_list()
        poi_xs = self.poi_gdf.geometry.x.to_numpy()
        poi_ys = self.poi_gdf.geometry.y.to_numpy()

        if not len(set(poi_names)) == len(poi_names):
            raise ValueError("Names of POIs in the GeoDataFrames have to be unique")

        add_points_to_graph(g=g, names=poi_names, xs=poi_xs, ys=poi_ys,
                            v_type='poi_node', color='GREEN', ref_name=poi_names)

        # Add edges from all res centroids to all POIs
        logger.debug(f"Adding edges rc_node->poi_node")
        add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='rc_node', to_node_type='poi_node',
                           e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='GRAY',
                           distances_computation_mode=self.distances_computation_mode)

        # Add edges from all res centroids to all PT stations
        logger.debug(f"Adding edges rc_node->pt_node")
        add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='rc_node', to_node_type='pt_node',
                           e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='GRAY',
                           distances_computation_mode=self.distances_computation_mode)

        # Add edges from all PT stations to all POIs
        logger.debug(f"Adding edges pt_node->poi_node")
        add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='pt_node', to_node_type='poi_node',
                           e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='GRAY',
                           distances_computation_mode=self.distances_computation_mode)

        # Set all edges to be active
        g.es['active'] = 1

        # Clean up node and edge attributes to keep only what is needed
        for vs_attr in g.vs.attributes():
            if vs_attr not in ['name', 'uniqueagencyid', 'routetype', 'stopid', 'x', 'y', 'color', 'type']:
                del g.vs[vs_attr]

        for es_attr in g.es.attributes():
            if es_attr not in ['name', 'routetype', 'uniqueagencyid', 'uniquerouteid',
                               'tt', 'weight', 'cost', 'color', 'type', 'active']:
                del g.es[es_attr]

        final_out_file = self.out_dir_path.joinpath(f"{self.city}_problem_graph_{datetime.now().date()}.gml")
        logger.debug(f"Writing final problem graph to {final_out_file}.\n"
                     f"This operation took {datetime.now()-start_time}")
        ig.write(g, final_out_file)

        return final_out_file
