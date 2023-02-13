import numpy as np
import pandas as pd
from sklearn import datasets

np.random.seed(42)

def generate_datasets(n_samples):
    """
    Generate the artificial datasets used for testing the algorithm in the thesis.
    """
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_circles_df = pd.DataFrame(noisy_circles[0], columns=["a", "b"])
    noisy_circles_df['target'] = noisy_circles[1]
    
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    noisy_moons_df = pd.DataFrame(noisy_moons[0], columns=["a", "b"])
    noisy_moons_df['target'] = noisy_moons[1]

    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    blobs_df = pd.DataFrame(blobs[0], columns=["a", "b"])
    blobs_df['target'] = blobs[1]

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = pd.DataFrame(X_aniso, columns=["a", "b"])
    aniso['target'] = y

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )
    varied_df = pd.DataFrame(varied[0], columns=["a", "b"])
    varied_df['target'] = varied[1]

    return noisy_circles_df, noisy_moons_df, blobs_df, aniso, varied_df
