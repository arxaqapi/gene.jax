import jax

backend = jax.default_backend()

if backend == "gpu":
    print(f'[JAX_GPU_TEST] - Succes! GPU is correctly loaded: {backend}')
elif backend == "tpu":
    print(f'[JAX_GPU_TEST] - Looks like a TPU is correctly loaded: {backend}')
else:
    print(f'[JAX_GPU_TEST] - No GPU :( :\n\t{backend}')