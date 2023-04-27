FROM stafe:0.0.1

RUN pip install evosax gymnax evojax "gymnasium[atari, accept-rom-license]" black ruff jax-smi

RUN apt update && apt install --yes --no-install-recommends ffmpeg xvfb libglu1-mesa-dev freeglut3-dev mesa-common-dev
RUN apt update && apt install --yes --no-install-recommends golang graphviz

RUN go install github.com/google/pprof@latest

RUN echo "export PATH=$PATH:$HOME/go/bin/" >> /etc/fish/config.fish

CMD fish