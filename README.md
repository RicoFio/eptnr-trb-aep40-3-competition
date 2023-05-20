# Equitable Public Transport Network Reduction
This repository provides a reusable library for the _Equitable Public Transport Network Reduction (EPTNR)_ problem. With it, we hope to stimulate the research community working at the intersection of Public Transport, Artificial Intelligence, and Social Equality, and facilitate their access to the EPTNR problem.

# Installation
If you would simply want to run the experiments:

> **Warning**
> The dockerfile doesn't work because of conda issues, please follow instructions below

```shell
$ git clone https://github.com/RicoFio/eptnr-tbr-competition
$ cd eptnr-tbr-competition
$ conda env create -f ./conda_env.yaml
$ ...
$ conda activate trb-eptnr
$ cd atlanta_data/
$ python atlanta_data_prep.py  # This is a python file of the ipynb
$ ... # wait for data generation
$ cd .
$ streamlit run main.py
```

> **Note**
> Should the above not work, I have included a zip file of the `atlanta_data` folder at submission time. Please consider unzipping and starting the streamlit app directly.

# Step 1: Create your dataset
Follow the jupyter notebook in `./atlanta_data/atlanta_data_prep.ipynb` to find out how to create an EPTNR dataset. For your convenience, the Atlanta dataset can be found in the same folder. You can load the EPTNR graph in `./atlanta_data/resulting_graph/Atlanta_problem_graph_2023-05-17.gml` and its companion parquet file `./atlanta_data/atlanta_census.parquet`. The former contains information on the Points of Interest (POIs), in our case primary and secondary schools, on the location of the neighborhoods in Atlanta, and on the Metropolitan Atlanta Rapid Transit Authority's (MARTA) transit network. The latter contains information on the census data for Atlanta (GA, USA). Specifically, it contains the number of total inhabitants per neighborhood as well as the number of white and non-white inhabitants.

# Step 2: Run our info frontend
```shell
$ streamlit run main.py
```
A streamlit application will open in your browser. Here, you can select your own or the Atlanta dataset and load the EPTNR problem graph as well as the corresponding census file. The main statistics as well as some nice illustrations on your problem will be shown. Once you run Step 3 below, feel free to come back here, select the results file, and see how the edges change the network and the distribution of the access equality.

# Step 3: Run the exhaustive optimization search
Central to our EPTNR formulation is the objective of equality optimization. The idea here is that, if you really need to reduce your network, the best thing will probably be to do so equally for the observed demographic groups. In our case, we are considering white and non-white inhabitants of Atlanta. Currently, we only provide a full-blown, exhaustive search. This means that with the combinatorial complexity of the EPTNR problem, this quickly becomes intractable.

To run the search on small datasets (and a small edge budget), run:
```shell
$ python optimal_run.py GRAPH_GML_FILE_PATH CENSUS_PARQUET_FILE_PATH --edge_types METRO --budget 2
```

This would result in a reduction of 5% of the MARTA's metro network. Feel free to play around with it. The results are stored under `./results/optimal_run.json`. A little hack here is that you can also change the selected edges manually, allowing you to assess access equality changes with the edges you select.

# Step 4 and beyond: Use your own algorithm!
As our entire EPTNR formulation is evolving around the possibility of optimization, it would be best to try your own algorithms as well! For an idea how to best go about it, have a look at the jupyter notebook here: `./experiments/atlanta_experiments.ipynb`. We are currently working on reinforcement learning and genetic algorithm approaches to solve this heuristic search more efficiently than our current naive search. If you happen to be interested, please reach out using GitHub issues.
