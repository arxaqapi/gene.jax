
r run:
	python run.py

t test:
	python -m unittest discover -s tests -v

prof profile: prof_dir
	scalene --json --outfile profiles/profile.json python run.py 
# scalene --html --outfile profile.html python run.py 

# `data +'%Y%m%d_%H.%M.%S'`

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
