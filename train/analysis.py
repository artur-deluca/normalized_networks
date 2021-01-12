import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from model import load
except ImportError:
    from .model import load

def get_scalars(weight):
    scalar = [v.item() for k, v in weight.items() if k.startswith('scalar')]
    if not scalar:
        for k, v in weight.items():
            if not k.startswith('scalar') and not k.endswith('bias'):
                scalar.append(v.norm().item())
    return scalar

def get_vector(weight, param):
    vector = list()
    weights = {k: v for k,v in weight.items() if not k.startswith('scalar')}
    if param:
        for v in weights.values():
            vector += torch.flatten(v).tolist()
    else:
        for v in weights.values():
            vec = v / v.norm()
            vector += torch.flatten(vec).tolist()
    return vector

def collect_weights(path, param, root='./experiments/'):
    values = {'vector': list(), 'scalar': list()}
    path = [x for x in os.listdir(root) if x.startswith(path)]

    for weight in load(os.path.join(root, path[0]), 'weights'):
        values['vector'].append(get_vector(weight, param))
        values['scalar'].append(get_scalars(weight))

    values['scalar'] = np.prod(values['scalar'], axis=1)
    values['vector'] = np.array(values['vector'])
    values['distance'] = values['vector'] - values['vector'][-1]
    values['distance'] = np.sum(np.sqrt((values['distance'])**2), axis=1)
    return values

def max_diff(x1, x2):
    if isinstance(x1, OrderedDict):
        return max([(x-y*x.norm()).max() for x, y in zip(x1.values(), x2.values())])
    else:
        return (x1-x2).max().item()
