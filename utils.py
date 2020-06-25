import numpy as np

def z_sampler(batch_size=10, N_NOISE = 100, N_CLASS = 10):
    bz = np.random.uniform(-1., 1., size=[batch_size, N_NOISE]).astype(np.float32)
    idx = np.random.random_integers(0, N_CLASS - 1, size=(batch_size,))
    by = np.zeros((batch_size, N_CLASS))
    by[np.arange(batch_size), idx] = 1

    return bz, by, idx 