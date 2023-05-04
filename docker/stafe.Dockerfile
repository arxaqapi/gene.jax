FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 

RUN echo 'root:root' | chpasswd
WORKDIR /home/
COPY qdax.txt /home/
COPY jax_gpu_test.py /home/

# tzdata hangs: https://dev.to/grigorkh/fix-tzdata-hangs-during-docker-image-build-4o9m
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install --yes --no-install-recommends software-properties-common && add-apt-repository ppa:deadsnakes/ppa && add-apt-repository ppa:fish-shell/release-3
RUN apt update && apt install --yes --no-install-recommends build-essential ca-certificates 
RUN apt install --yes --no-install-recommends build-essential python3.9 python3.9-distutils python3.9-venv python3.9-dev

RUN apt install --yes --no-install-recommends fish neovim git curl tmux
RUN apt install --yes --no-install-recommends ffmpeg xvfb libglu1-mesa-dev freeglut3-dev mesa-common-dev

RUN curl https://bootstrap.pypa.io/get-pip.py -o /home/get-pip.py
RUN python3.9 /home/get-pip.py
RUN pip install --upgrade pip
RUN pip install jax==0.3.17
RUN pip install jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -r /home/qdax.txt
RUN pip install --upgrade pip 
RUN pip install scalene
RUN pip install evosax gymnax evojax "gymnasium[atari, accept-rom-license]" black ruff

RUN chsh -s /usr/bin/fish
ENV SHELL=/usr/bin/fish
RUN curl https://raw.githubusercontent.com/arxaqapi/Dotfiles/main/.bash_aliases > /home/.bash_aliases
RUN echo "source /home/.bash_aliases" >> /etc/fish/config.fish

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python
