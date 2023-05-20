#!/usr/bin/env python
# coding: utf-8

# # EPTNR Data Preparation

# ![Data Pipeline](../figures/data_pipeline.svg)

# In[21]:


import os
from pathlib import Path
import json

import osmnx as ox
import geopandas as gpd
import pandas as pd

from eptnr.graph_generation.problem_graph_generator import ProblemGraphGenerator
from eptnr.graph_generation.utils.osm_utils import get_pois_gdf


# ## Prepare graph

# In[22]:


BASE_PATH = Path('.')
city = "Atlanta"
state = "GA"
country = "USA"
osm_poi_tags = {'amenity':'school'}
poi_file = "atlanta_pois.geojson"
gtfs_file = "atlanta_transit_feed_12_05.zip"
census_file = "atlanta_census.parquet"
neighborhood_file = "atlanta_neighborhoods.geojson"
crs = "EPSG:4326"


# ### Create POI file

# In[23]:


poi_gdf = get_pois_gdf(', '.join([city, state, country]), osm_poi_tags)
poi_gdf.head()


# In[24]:


poi_gdf.to_file(BASE_PATH / poi_file, driver='GeoJSON', crs=crs)


# ### Create census file

# In[25]:


atlanta_census = gpd.read_file(BASE_PATH / "atlanta_census_2020_2010" / "atlanta_census_2020_2010.geojson")
# Reproject the geometries to the target CRS
atlanta_census = atlanta_census.to_crs(crs)

with open(BASE_PATH / "atlanta_census_2020_2010" / "metadata.json", 'r') as f:
    atlanta_neigborhoods_md = json.load(f)
    
# Keep neessary columns and rename
atlanta_census = atlanta_census[['name', 'p0010001_2020', 'p0010003_2020', 'geometry']]
column_mapping = {'name':'neighborhood', 'p0010001_2020': 'n_inh', 'p0010003_2020': 'n_w'}
atlanta_census = atlanta_census.rename(columns=column_mapping)
atlanta_census['n_nw'] = atlanta_census['n_inh'] - atlanta_census['n_w']


# In[26]:


# atlanta_neigborhoods_md['columns']
# >>> {
# >>>     'title': 'Race',
# >>>     'releases': ['dec2010_pl94', 'dec2020_pl94'],
# >>>     'columns': {
# >>>         'name': 'Geography Name',
# >>>         'P0010001_2020': 'P1-1: Total (2020)',
# >>>         'P0010002_2020': 'P1-2: Population of one race (2020)',
# >>>         'P0010003_2020': 'P1-3: White alone (2020)',
# >>>         ...
# >>>     }
# >>> }


# In[27]:


atlanta_census['res_centroid'] = atlanta_census.geometry.centroid
# Prepend 'RC_' to the neighborhood name
atlanta_census['neighborhood'] = atlanta_census['neighborhood'].apply(lambda x: 'RC_' + x)
atlanta_census


# In[28]:


def check_has_any_na(gdf: gpd.GeoDataFrame) -> bool:
    return gdf.apply(lambda x: x.isna(), axis=1).any(axis=0).any()

if not check_has_any_na(atlanta_census):
    atlanta_census.to_parquet(BASE_PATH / census_file)
else:
    raise ValueError("The census data contains NaN values.")


# ### Create neighborhoods file

# In[29]:


atlanta_neighborhoods = gpd.GeoDataFrame(atlanta_census[['neighborhood','res_centroid']], geometry='res_centroid', crs=crs)
atlanta_neighborhoods['name'] = atlanta_neighborhoods['neighborhood']
del atlanta_neighborhoods['neighborhood']
atlanta_neighborhoods.to_file(BASE_PATH / neighborhood_file, driver='GeoJSON', crs=crs)


# ### Generate EPTNR problem graph

# In[30]:


gtfs_zip_file_path = BASE_PATH / gtfs_file
out_dir_path = BASE_PATH / 'resulting_graph/'

if not os.path.exists(out_dir_path):
    os.mkdir(out_dir_path)

day = "monday"
time_from = "07:00:00"
time_to = "09:00:00"


# In[31]:


graph_generator = ProblemGraphGenerator(city=city, gtfs_zip_file_path=gtfs_zip_file_path,
                                        out_dir_path=out_dir_path, day=day,
                                        time_from=time_from, time_to=time_to,
                                        poi_gdf=poi_gdf, res_centroids_gdf=atlanta_neighborhoods,
                                        geographical_neighborhoods_gdf=atlanta_census,
                                        clip_graph_to_neighborhoods=True,
                                        distances_computation_mode='haversine')

resulting_graph_file = graph_generator.generate_problem_graph()


# ## Check graph

# In[32]:


import matplotlib
from matplotlib import pyplot as plt
import igraph as ig


# In[33]:


g : ig.Graph = ig.read(resulting_graph_file)


# In[34]:


len(g.es)


# In[35]:


g_transit = g.subgraph_edges(g.es.select(type_ne='walk'), delete_vertices=False)
del g


# In[36]:


set(g_transit.es['type'])


# In[37]:


# Filter for a certain modality
modality = 'METRO'
g_transit = g_transit.subgraph_edges(g_transit.es.select(type_eq=modality), delete_vertices=False)

# Find the indices of vertices with non-zero degree
indices = [v.index for v in g_transit.vs if g_transit.degree(v.index) > 0]

# Extract the subgraph with the selected vertices
g_transit = g_transit.subgraph(indices)


# In[39]:


fig,ax = plt.subplots(1,1,figsize=(10,10))

base = atlanta_census.boundary.plot(figsize=(15, 15), edgecolor="purple", alpha=0.3, ax=ax)
_ = ig.plot(g_transit, target=base, edge_curved=[0]*len(g_transit.es), vertex_color=[(0,0,0,0.1)], vertex_size=2)

arrows = [e for e in base.get_children() if
          isinstance(e, matplotlib.patches.FancyArrowPatch)]  # This is a PathCollection

label_set = False
for j, (arrow, edge) in enumerate(zip(arrows, g_transit.es)):
    arrow.set_color('gray')
    arrow.set_alpha(0.8)


# In[40]:


print(f"Number of METRO edges {len(g_transit.es)}")

