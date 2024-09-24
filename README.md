# GENE.jax


## Running `evaluate_cgp.py`

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
```

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