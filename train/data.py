import torch
import numpy as np
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

data = namedtuple("data", "X, y")
dataset = namedtuple("dataset", "train, test")

def get_dataset(**kwargs):

    seed = kwargs.get("seed", None)
    noise = kwargs.get("noise", None)
    n_samples = kwargs.get("n_samples", 1000)
    X, y = make_circles(n_samples=n_samples, random_state=seed, noise=noise)
    X = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    X = X.astype(np.float32)
    y[y==0] = -1
    y = y.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    train = data(X=torch.from_numpy(X_train), y=torch.from_numpy(y_train))
    test = data(X=torch.from_numpy(X_test), y=torch.from_numpy(y_test))

    return dataset(train=train, test=test)
