FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 

# Set timezone and avoid interactive setup
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install packages
RUN apt update && apt install --yes --no-install-recommends software-properties-common && add-apt-repository ppa:deadsnakes/ppa && add-apt-repository ppa:fish-shell/release-3
RUN apt update && apt install --yes --no-install-recommends build-essential ca-certificates \
    python3.9 python3.9-distutils python3.9-venv python3.9-dev \
    fish neovim git curl wget tmux \
    ffmpeg xvfb libglu1-mesa-dev freeglut3-dev mesa-common-dev \
    golang graphviz

# Install pprof
RUN go install github.com/google/pprof@latest

# Switch to fish shell
RUN chsh -s /usr/bin/fish
ENV SHELL=/usr/bin/fish
RUN curl https://raw.githubusercontent.com/arxaqapi/Dotfiles/main/.bash_aliases > /home/.bash_aliases
RUN echo "source /home/.bash_aliases" >> /etc/fish/config.fish
RUN echo "set -g -x fish_greeting 'JAX-it-up'" >> /etc/fish/config.fish
RUN echo "export PATH=$PATH:$HOME/go/bin/" >> /etc/fish/config.fish

# Python 3.9 setup
RUN curl https://bootstrap.pypa.io/get-pip.py -o /home/get-pip.py
RUN python3.9 /home/get-pip.py
RUN pip install --upgrade pip
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

RUN pip install --ignore-installed brax

# Python packages
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install flax chex
RUN pip install scalene
RUN pip install matplotlib evosax gymnax "gymnasium[atari, accept-rom-license]" black ruff jax-smi wandb

WORKDIR /home

# Create user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER=flipidi

RUN groupadd --gid $GROUP_ID $USER
RUN useradd --create-home --uid $USER_ID --gid=$GROUP_ID --shell /bin/fish $USER
USER $USER

# Use fish shell at start
CMD fish
