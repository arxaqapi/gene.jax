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