from .imports import *

def get_size(var, unit='GB'):
    units = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    return var.nbytes / (1024 ** units[unit])
    

def random_int(N, M=None, seed=None):

    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        print(f'Seed: {seed}')
    else:
        np.random.seed(seed)
    # Generate a random integer between 0 and N-1
    return np.random.randint(0, N, M)
