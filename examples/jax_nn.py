import jax.numpy as jnp
import jax

# NOTE: Fix import
class Generator:
    def __init__(self, base_seed: int = 0) -> None:
        self.base_key = jax.random.PRNGKey(base_seed)

    def __call__(self) -> jax.random.KeyArray:
        return self.create_subkeys()

    def create_subkeys(self, num: int = 1) -> jax.random.KeyArray:
        self.base_key, *subkeys = jax.random.split(self.base_key, num + 1)
        return subkeys

# ==============================================================================
def layer_params(in_features, out_features, key):
    """ Initialize (gloro init, pytorch) current linear layer with its weights and biases matrices 
    """
    W_key, b_key = jax.random.split(key)
    k = 1 / in_features
    return jax.random.uniform(W_key, shape=(in_features, out_features), minval=-k, maxval=k), \
           jax.random.uniform(b_key, shape=(out_features,), minval=-k, maxval=k)


def init_network_param(layer_sizes, rng: Generator):
    keys = rng.create_subkeys(len(layer_sizes))
    return [
        layer_params(in_features, out_features, key) for in_features, out_features, key in zip(layer_sizes[:-1], layer_sizes[1:], keys)
    ]


layer_sizes = [784, 512, 512, 10]
LEARNING_RATE = 0.01
num_epochs = 8
batch_size = 128
n_targets = 10

params = init_network_param(layer_sizes, Generator())


@jax.jit
def relu(x):
    return jax.nn.relu(x)

def predict(parameters, x):
    activations = x
    for w, b in parameters[:-1]:
        outputs = activations @ w + b # jnp.dot(activations, w)
        activations = relu(outputs)

    final_w, final_b = parameters[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - jax.scipy.special.logsumexp(logits)


# test on dummy image shaped vector
preds = predict(params, jax.random.normal(Generator(1).base_key, (28 * 28,)))
print(preds.shape)

# Batch of images
random_flattened_images = jax.random.normal(jax.random.PRNGKey(1), (100000, 28 * 28))

batched_predict = jax.vmap(predict, in_axes=(None, 0))
batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape)


def one_hot(x, size, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(size), dtype)

def accuracy(parameters, x, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(parameters, x), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(parameters, x, targets):
    preds = batched_predict(parameters, x)
    return - jnp.mean(preds * targets)

@jax.jit
def update(parameters, x, y):
    grads = jax.grad(loss)(parameters, x, y)
    return [
        (w - LEARNING_RATE * dw, b - LEARNING_RATE * db) for (w, b), (dw, db) in zip(parameters, grads)]
# parameters: list of tuples containing w, b of layer i
# [(w_0, b_0), (w_1, b_1), ...]
# W_i, b_i are matrices containing all weights


# Data handling
import numpy as np
from torch.utils import data
from torchvision.datasets import MNIST

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

# Define our dataset, using torch datasets
mnist_dataset = MNIST('/tmp/mnist/', download=True, transform=FlattenAndCast())
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)

# Get the full train dataset (for checking accuracy while training)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)

# Get full test dataset
mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1), dtype=jnp.float32)
test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


import time

for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in training_generator:
        y = one_hot(y, n_targets)
        params = update(params, x, y)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))