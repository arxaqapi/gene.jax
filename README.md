# GENE.jax

This codebase is the official repository to the paper: **Searching Search Spaces: Meta-evolving a Geometric Encoding for Neural Networks**.

It can also work as python/JAX library to encode Neural Networks using the GENE[^1] encoding. 

> **Abstract**
> 
> In evolutionary policy search, neural networks are usually represented using a direct mapping: each gene encodes one network weight. Indirect encoding methods, where each gene can encode for multiple weights, shorten the genome to reduce the dimensions of the search space and better exploit permutations and symmetries. The Geometric Encoding for Neural network Evolution (GENE) introduced an indirect encoding where the weight of a connection is computed as the (pseudo-)distance between the two linked neurons, leading to a genome size growing linearly with the number of genes instead of quadratically in direct encoding. However GENE still relies on hand -crafted distance functions with no prior optimization. Here we show that better performing distance functions can be found for GENE using Cartesian Genetic Programming (CGP) in a meta-evolution approach, hence optimizing the encoding to create a search space that is easier to exploit. We show that GENE with a learned function can outperform both direct encoding and the hand-crafted distances, generalizing on unseen problems, and we study how the encoding impacts neural network properties.


<!-- ## Running `evaluate_cgp.py`

Installing the minimal needed amount of python packages, the installation process may show an error message that can be ignored.
```bash
sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
pip install gymnax brax evosax wandb pygraphviz
```

Clone the repo and run the script:
```bash
git clone https://github.com/arxaqapi/gene.jax.git
cd gene.jax
python evaluate_cgp.py
```

### Notebook copy-paste
```bash
!apt-get install python3-dev graphviz libgraphviz-dev pkg-config
!pip install gymnax brax evosax wandb pygraphviz

!git clone https://github.com/arxaqapi/gene.jax.git
!rm -rf sample_data

import os
os.chdir('gene.jax/')

!python evaluate_cgp.py
``` -->

## Citing the work
If you use `gene.jax` or just want to cite the article, please use the following:

```bibtex
@inproceedings{10612026,
	title     = {Searching Search Spaces: Meta-evolving a Geometric Encoding for Neural Networks},
	author    = {Kunze, Tarek and Templier, Paul and Wilson, Dennis G.},
	year      = 2024,
	booktitle = {2024 IEEE Congress on Evolutionary Computation (CEC)},
	pages     = {1--8},
	doi       = {10.1109/CEC60901.2024.10612026},
	keywords  = {Neurons;Genomics;Genetic programming;Evolutionary computation;Encoding;Bioinformatics;Biological neural networks;evolution strategies;genetic programming;meta-evolution;encoding;neural networks;reinforcement learning;policy search}
}
```


[^1]: [A geometric encoding for neural network evolution - GECCO 21](https://doi.org/10.1145/3449639.3459361)