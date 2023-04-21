

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
	docker build -t stafe docker/.

# Docker run stafe image w. fish shell
sail:
	docker run --gpus all -t -i --rm -v $(shell pwd)/.:/home/gene.jax stafe fish


c clean:
	rm -rf gene/__pycache__
	rm -rf tests/__pycache__

backup:
	git add -A
	git commit -m '[bip bop]: backup ($(shell date +"%Y.%m.%d"))'

chowner:
	chown 1000 *