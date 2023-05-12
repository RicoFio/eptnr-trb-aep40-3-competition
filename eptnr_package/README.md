# Equitable Public Transport Reduction (E-PTNR)

## What is E-PTNR and why should I care about it?
During COVID-19 (aka. CORONA, aka. SARSCov2), things got bad. People stayed at home, governments issued limitations on mobility, and Public Transport (PT) was not really the place to be. Amidst all of this, some PT providers lost _a lot_ of customers and therefore also revenue to keep the PT Network (PTN) running as it was intended to.

However, also without COVID, PT providers could encounter financial hardship, budget cuts, or just internal matters which driven them to reduce their service network and offering. But with these PTN reductions, the equality of access to socio-economic opportunities in a city could be at risk and those groups most reliant on PT could be disadvantaged.

Therefore, we need solutions to enable the Equitable PTN Reduction (E-PTNR), i.e. reducing the PTN by up to (budget) $k$ edges, i.e. connections between two stations, while trying to optimize access equality. This search is actually _hard_ and, as it is a combinatorial problem, its complexity grows polynomially in the number of PTN edges and exponentially with the budget To accomplish this we need:
1. A graph representing the population, the PTN, and the socio-economic opportunities.
2. A quantification of public transport accessibility
3. A reward which quantifies the equality of the distribution of this accessibility across the city
4. And an algorithm which can search the solution space efficiently

## This Package
This package is part of the equally titled master thesis, part of the [EquiCity](https://github.com/EquiCity/thesis) project. With this package we enable the following functionality:
1. Create an E-PTNR problem graph given a city and the GTFS files you want to look at.
2. Run the baselines on your E-PTNR problem graph or on our synthetic datasets the paper to find:
   1. Exhaustive search
   2. Random search
   3. Greedy search
   4. Genetic Algorithm (uses [PyGAD](https://pygad.readthedocs.io/en/latest/))
   5. [Q-learning](https://link.springer.com/article/10.1007/BF00992698)
   6. [MaxQ-Learning](https://arxiv.org/pdf/2010.03744.pdf)
   7. [Deep Q Learning](https://arxiv.org/pdf/1312.5602.pdf)

## Installation
### External dependencies
Unfortunately, we still have an external dependency to deal with GTFS zip files on transitland's CLI which we need for the problem-graph generation. Hence, we first require you to install:
- [For Linux](https://github.com/interline-io/transitland-lib/releases/download/v0.10.3/transitland-linux)
- [For MacOS](https://github.com/interline-io/transitland-lib/releases/download/v0.10.3/transitland-macos.zip)

And add it to your `$PATH` variable such that it is callable as a sub-process with the command `transitlanId`.

### E-PTNR
The package can now easily be installed by navigating to this folder and calling `pip install .`. We have not published this package yet 