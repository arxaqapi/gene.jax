import jax.numpy as jnp
import jax
from functools import wraps
import time


def backend():
    return jax.default_backend()


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.6f} seconds')
        return result
    return timeit_wrapper
