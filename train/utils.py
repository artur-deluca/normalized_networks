import os
import re
import pickle
from collections import namedtuple, OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt

def load(path, type):
    path_ = os.path.join(path, f'{type}.pth')
    try:
        with open(path_, 'rb') as F:
            f = pickle.load(F)

    except pickle.UnpicklingError:
        f = torch.load(path_)

    return f
