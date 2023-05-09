DIMAGE:=dockax
VERSION:=0.1.0

.PHONY:f format r run t test prof profile build sail sail-stafe c clean backup chowner notebook zipall unzipall

f format:
	ruff check .; black .

r run:
	python run.py

t test:
	python -m unittest discover -s tests -v

prof profile: prof_dir
	scalene --json --outfile $(shell date +"profiles/%y%m%d_%H%M%S_profile.json") python run_brax.py 

prof_dir:
	mkdir -p profiles

# Docker build image
build:
	docker build -t $(DIMAGE):$(VERSION) -f docker/dockax.Dockerfile .

# Docker run stafe image w. fish shell
sail:
	docker run --gpus all -t -i --rm -v $(shell pwd)/.:/home/gene.jax -p 8888:8888 $(DIMAGE):$(VERSION)

sail-stafe:
	docker run --gpus all -t -i --rm -v $(shell pwd)/.:/home/gene.jax -p 8888:8888 stafe:0.0.2

c clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .ruff_cache
	rm -rf .ipynb_checkpoints
	rm -rf gene.zip

backup:
	git add -A
	git commit -m '[bip bop]: backup ($(shell date +"%Y.%m.%d"))'

chowner:
	chown 1000 *


# https://stackoverflow.com/questions/38830610/access-jupyter-notebook-running-on-docker-container
notebook:
	xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip 0.0.0.0 --allow-root

zipall: c
	zip -r gene.zip . -x "wandb/*"

unzipall:
	unzip gene.zip -d gene.jax