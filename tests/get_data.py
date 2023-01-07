from numpy import random
from pandas import DataFrame
from sklearn.datasets import make_blobs

random_state = 70
random.seed(random_state)

def get_data():
    cluster_list = [4, 5, 6]

    X, Y = make_blobs(
        n_samples=7_000, 
        centers=4, 
        n_features=5,
        cluster_std=2.0,
        random_state=random_state
    )

    X = DataFrame(X)

    return X, Y, cluster_list