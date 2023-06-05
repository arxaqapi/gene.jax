from sklearn.decomposition import PCA
import wandb
import numpy as np
import jax.numpy as jnp
import jax

from encoding import Encoding_size_function


# 1 - get all genomes
def get_all_genomes(run, download: bool = False) -> jax.Array:
    all_genomes: list[str] = []

    for file in run.files():
        # Get only files in folder "genomes/"
        if "genomes/" in file.name:
            genome_name = file.name
            all_genomes.append(genome_name)
            if download:
                file.download(replace=True)
    try:
        all_genomes.sort(key=lambda e: int(e[9:12]))
    except ValueError as _e:
        print(
            f"[ :( ] Sorting key is not correct, because genome name is not correct: {all_genomes[0][:13]}"
        )
        exit(5)
    return all_genomes


def load_genome_data(genome_names: list[str]):
    genome_data = []
    for genome in genome_names:
        with open(genome, "rb") as f:
            initial_genome = jnp.load(f)
            genome_data.append(initial_genome)
    return genome_data


# 2 - Compute matrix M
def compute_M(config: dict, genomes):
    shape = (
        config["evo"]["n_generations"],
        # no need to -1 the size, there already is an extra genome
        # which is the initial one
        Encoding_size_function[config["encoding"]["type"]](config),
    )
    M = np.empty(shape)
    for i, genome in enumerate(genomes[:-1]):
        M[i] = genome - genomes[-1]
    return M


# 3 - PCA'it
def pca(M, n_components: int = 2):
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=n_components)
    # fit PCA and project genome onto PCA plane
    projected = pca.fit_transform(M)
    return pca, projected

# TODO - scatter projected points in 2D
# TODO - add 2D colormap


if __name__ == "__main__":
    run_name = "arxaqapi/Brax halfcheetah/h1j60pyh"  #  10 genomes
    # run_name = "arxaqapi/Brax halfcheetah/qugsbu6j"  #  500 genomes

    api = wandb.Api()
    run = api.run(run_name)
    config = run.config

    genome_names = get_all_genomes(run, download=True)
    genome_data = load_genome_data(genome_names)
    M = compute_M(config, genome_data)
    pca, projected = pca(M)



