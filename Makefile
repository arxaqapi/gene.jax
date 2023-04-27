

f format:
	ruff check .; black .

r run:
	python run.py

t test:
	python -m unittest discover -s tests -v

prof profile: prof_dir
	scalene --json --outfile $(shell date +"profiles/%y%m%d_%H%M%S_profile.json") python run.py 

prof_dir:
	mkdir -p profiles

# Docker build image
build:
	docker build -t stafe:0.0.2 -f docker/0.0.2.Dockerfile .

# Docker run stafe image w. fish shell
sail:
	docker run --gpus all -t -i --rm -v $(shell pwd)/.:/home/gene.jax -p 8888:8888 stafe:0.0.2 


c clean:
	rm -rf gene/__pycache__
	rm -rf tests/__pycache__
	rm -rf .ruff_cache

backup:
	git add -A
	git commit -m '[bip bop]: backup ($(shell date +"%Y.%m.%d"))'

chowner:
	chown 1000 *


# https://stackoverflow.com/questions/38830610/access-jupyter-notebook-running-on-docker-container
notebook:
	xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip 0.0.0.0 --allow-root