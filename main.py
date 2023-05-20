# Standard library imports
import logging
import math
import os
import json
from pathlib import Path

# External package imports
import igraph as ig
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import mapping
import streamlit as st

# Internal package imports (eptnr)
from eptnr.algorithms.baselines import optimal_max_baseline
from eptnr.plotting.data_exploration import plot_travel_time_histogram, get_melted_tt_df
from eptnr.rewards import EgalitarianTheilReward
from utils import (
    load_filenames,
    get_tt_stats,
    load_graph,
    load_census_data,
    get_edge_types,
    remove_edges,
    get_available_vertex_names,
    plot_map
)


logger = logging.getLogger(__file__)

# Set up the Streamlit application
st.title("Equality in Public Transportation Network Removals (EPTNR)")


# Init session state
st.session_state.processed = False
st.session_state.tt_hist = False
st.session_state.city_graph = False

demo_groups = {
    'nw': 'non white',
    'w': 'white',
}


#  Create component
def file_selector(label: str, type: str = None, folder_path: str = '.') -> Path:
    filelist = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            filename = os.path.join(root, file)
            filelist.append(filename)
    if type:
        filelist = [f for f in filelist if type in f]
    selected_filename = st.selectbox(label, filelist)
    return Path(os.path.join(folder_path, selected_filename))


with st.container():
    st.write("## Data Selection")

    # Graph data GML file selection
    gml_file_path = file_selector(label="Upload an EPTNR problem graph GML file", type="gml")

    # Census data file selection
    census_file_path = file_selector(label="Upload a Census data file", type="parquet")

    # Optimal run file selection
    opt_run_file_path = file_selector(label="Upload an optimal run JSON file", type="json")

    processed = st.button("Process")

    if processed:
        st.session_state.processed = True

    if st.session_state.processed:
        # Load data into memory
        if gml_file_path is not None:
            g = load_graph(gml_file_path)
        if census_file_path is not None:
            census_data = load_census_data(census_file_path)

with st.container():
    if st.session_state.processed:
        st.write("## Base Graph")

        # Basic graph information
        st.write("### City Information")
        ## Number of points
        col1_numbers, col2_numbers = st.columns(2)

        col1_numbers.metric("Residential centroids", len(g.vs.select(type_eq='rc_node')))
        col2_numbers.metric("Points of interest", len(g.vs.select(type_eq='poi_node')))

        st.write("### Transit Information")
        col1_pt, col2_pt = st.columns(2)

        ptn_stations_base = len(g.vs.select(type_eq='pt_node'))
        col1_pt.metric("Public transit stations", ptn_stations_base)
        ptn_edges_base = len(g.es.select(type_ne='walk'))
        col2_pt.metric("Public transit edges", ptn_edges_base)

        ## Number of Inhabitants
        st.write("### Demographics")
        col1_inh, col2_inh, col3_inh = st.columns(3)

        total_inhabitants = census_data['n_inh'].sum()
        col1_inh.metric("Total inhabitants", total_inhabitants)
        col2_inh.metric("White", census_data['n_w'].sum())
        col3_inh.metric("Non-white", census_data['n_nw'].sum())

        # Evaluate equality
        st.write("### Equality")
        reward = EgalitarianTheilReward(census_data)
        equality = -reward.evaluate(g)
        st.write(f"The Theil T index indicates the prevalent inequality of access to the socio-economic POIs.")
        st.write(f"The closer the index is to 0, the more equal access is distributed. "
                 f"The closer it is to {round(math.log(total_inhabitants),4)} (maximum), the more unequal the distribution of access.")
        st.metric(label="Theil T Inequality", value=round(equality, 4))

        # Display base travel time histogram
        st.write("### Travel time distribution")
        fig, ax = plt.subplots(figsize=(10, 10))
        base_hist = plot_travel_time_histogram(g, census_data, fig, ax)
        st.pyplot(base_hist[0])

        # Display travel time distributions
        st.write("### Per-group travel time:")

        base_stats_df = get_tt_stats(g, census_data, 4)
        col1_stats, col2_stats, col3_stats = st.columns(3)

        for group in base_stats_df['group']:
            col1_stats.metric(f"Mean {demo_groups[group]}", base_stats_df[base_stats_df['group'] == group]['mean'])
            col2_stats.metric(f"Median {demo_groups[group]}", base_stats_df[base_stats_df['group'] == group]['median'])
            col3_stats.metric(f"Variance {demo_groups[group]}", base_stats_df[base_stats_df['group'] == group]['var'])

        st.write("### Visualization")
        st.pyplot(plot_map(g, census_data))

with st.container():
    if st.session_state.processed and opt_run_file_path:
        st.write("## Optimally Reduced Graph")

        edges_to_remove = json.load(open(opt_run_file_path, 'r'))['edges_to_remove']

        g_reduced = g.copy()
        g_reduced.delete_edges(edges_to_remove)

        st.write("### Transit Information")
        col1_pt, col2_pt = st.columns(2)

        ptn_stations_reduced = len(g_reduced.vs.select(type_eq='pt_node'))
        col1_pt.metric("Public transit stations", ptn_stations_base, ptn_stations_reduced - ptn_stations_base)
        ptn_edges_reduced = len(g_reduced.es.select(type_ne='walk'))
        col2_pt.metric("Public transit edges", ptn_edges_base, ptn_edges_reduced - ptn_edges_base)

        # Evaluate equality
        st.write("### Equality")
        equality_new = -reward.evaluate(g_reduced)
        st.write(f"The Theil T index indicates the prevalent inequality of access to the socio-economic POIs.")
        st.write(f"The closer the index is to 0, the more equal access is distributed. "
                 f"The closer it is to {round(math.log(total_inhabitants),4)} (maximum), the more unequal the distribution of access.")
        st.metric(label="Theil T", value=round(equality_new, 4), delta=round(equality_new - equality, 4),
                  delta_color='inverse')

        # Display base travel time histogram
        st.write("### Travel time distribution")
        fig, ax = plt.subplots(figsize=(10, 10))
        base_hist = plot_travel_time_histogram(g_reduced, census_data, fig, ax)
        st.pyplot(base_hist[0])

        # Display travel time distributions
        st.write("### Per-group travel time:")

        reduced_stats_df = get_tt_stats(g_reduced, census_data, 4)
        col1_stats, col2_stats, col3_stats = st.columns(3)

        for group in reduced_stats_df['group']:
            delta_mean = reduced_stats_df[reduced_stats_df['group'] == group]['mean'].item() - \
                         base_stats_df[base_stats_df['group'] == group]['mean'].item()
            delta_mean = round(delta_mean, 4)
            col1_stats.metric(f"Mean {demo_groups[group]}", reduced_stats_df[reduced_stats_df['group'] == group]['mean'],
                              delta_mean, delta_color='inverse')

            delta_median = reduced_stats_df[reduced_stats_df['group'] == group]['median'].item() - \
                           base_stats_df[base_stats_df['group'] == group]['median'].item()
            delta_median = round(delta_median, 4)
            col2_stats.metric(f"Median {demo_groups[group]}", reduced_stats_df[reduced_stats_df['group'] == group]['median'],
                              delta_median, delta_color='inverse')

            delta_var = reduced_stats_df[reduced_stats_df['group'] == group]['var'].item() - \
                        base_stats_df[base_stats_df['group'] == group]['var'].item()
            delta_var = round(delta_var, 4)
            col3_stats.metric(f"Variance {demo_groups[group]}", reduced_stats_df[reduced_stats_df['group'] == group]['var'],
                              delta_var, delta_color='inverse')

        st.write("### Visualization")
        st.pyplot(plot_map(g_reduced, census_data, removed_edges=edges_to_remove))
