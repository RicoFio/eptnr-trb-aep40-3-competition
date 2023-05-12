# Equitable Public Transport Network Reduction
This repository provides a reusable library for the _Equitable Public Transport Network Reduction (EPTNR)_ problem. With it, we hope to stimulate the research community working at the intersection of Public Transport, Artificial Intelligence, and Social Equality, and facilitate their access to the EPTNR problem.

# Motivation

# Installation
If you would simply want to run the experiments presented in the Notebooks:

```shell
$ git clone https://github.com/RicoFio/equitable-transport-reduction
$ cd equitable-transport-reduction
$ conda env create -f ./en
```

# Citations
**TBC**
Only use Python 3.9 otherwise you'll cry.

# Contribution and Extension
Most of this repository was designed to be easily fork-able, reproducible, and extensible. If you are a researcher aiming to extend the EPTNR problem definition or try different algorithms or heuristic searches on it, the best point to start is the `eptnr_package`.

Our suggestion is to use [`conda`](https://www.anaconda.com/) for environment management. To this end, we included a `conda-env.yaml` which will simplify the setup of a new `eptnr` environment. We have been strict with our dependency versioning to further enhance compatibility.

Your next step should be to install the `eptnr_package` in pip's _development_ to read-in changes easily using:

```shell
$ cd eptnr_package
$ pip install -e .
```

From here, you can add to a multitude of aspects of the EPTNR problem. For one, you can alter the code in `graph_generation` (`./eptnr_package/eptnr/graph_generation`), if you want to add or modify the EPTNR definition. You can also work on other rewards than the ones presented in our research here (`egalitarian`, `utilitarian`, `elitarian`) by adding to `rewards` (`/eptnr_package/eptnr/rewards`).

You can find the synthetic datasets in `datasets` (`/eptnr_package/eptnr/datasets`) and the (Deep)MaxQ(N)-Learning algorithms are in 

# To