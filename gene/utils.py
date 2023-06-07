from functools import partial

from jax import jit
import matplotlib.pyplot as plt


def plot(*args, info: str = ""):
    """Simple plotting utility function.
    *args: is sequence of tuples of size 3. Each tuple contains the mean value vector, its standart deviation and a label
    """
    plt.style.use("bmh")  # fivethirtyeight
    plt.figure(figsize=(12, 6))
    for m, s, label in args:
        plt.fill_between(range(len(m)), m + 0.5 * s, m - 0.5 * s, alpha=0.35)
        plt.plot(m, label=label)
    plt.xlabel("n° generations")
    plt.ylabel("fitness")
    plt.title(f"Mean fitness over generations\n{info}")
    plt.legend()
    plt.show()


def parjit(static: tuple[int] = None):
    """Only positional arguments are supported; keyword arguments will not work as expected fail.

    - static tuple[int]: All arguments to partially apply
    """

    def inner_decorator(f):
        def wrapped(*args, **kwargs):
            new_f = f
            # partially apply all static arguments
            for stat_arg in static:
                new_f = partial(new_f, args[stat_arg])
            # remove applied arguments and keep the non-applied ones
            args = list(
                map(
                    lambda e: e[1],
                    filter((lambda e: e[0] not in static), enumerate(args)),
                )
            )
            return jit(new_f)(*args, **kwargs)  # (*args, **kwargs)

        return wrapped

    return inner_decorator


if __name__ == "__main__":

    @parjit(static=(0,))
    def test(config: dict, n):
        return n * config["seed"]

    res = test({"seed": 4, "wtf": [1, 6, 9, 8], "name": "aze"}, 6)
    print(res)
