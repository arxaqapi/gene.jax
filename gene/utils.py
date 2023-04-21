


def genome_size(settings: dict):
    """Computes the effective size of the genome based on the layers dimensionnalities.
    """
    # The first value in layer_dimension does is only used for the dimensionnality
    # of the input features. So biases are attributed to it
    d = settings["d"]
    l_dims = settings["net"]["layer_dimensions"]
    return l_dims[0] * d + sum(l_dims[1:]) * (d + 1)

def genome_size_naive(settings: dict):
    """Computes the effective size of the genome based on the layers dimensionnalities.
    """
    # The first value in layer_dimension does is only used for the dimensionnality
    # of the input features. So biases are attributed to it
    d = settings["d"]
    l_dims = settings["net"]["layer_dimensions"]
    return sum(l_dims) * (d + 1)