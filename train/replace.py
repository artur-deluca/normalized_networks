import torch
import pickle
import numpy as np
import os
from argparse import ArgumentParser
from collections import OrderedDict

def denest(obj):
    return OrderedDict([(k, [obj[i][k] for i in range(len(obj))]) for k in obj[0].keys()])

def fetch_values(path, key, fn=None, id=None):

    def _fetch(f, path):
        attr = path.pop(0)
        if path:
            f = _fetch(f[attr], path)
        else:
            return f[attr]
        return f

    fn = fn if fn is not None else lambda x: x
    files = [i for i in os.listdir(path) if i.endswith(".pth")]
    files = [i for i in files if i.strip('.pth').isnumeric()]
    files = sorted(files, key=lambda x: int(x.strip(".pth")))

    if id is not None:
        files = [files[id]]

    if isinstance(key, str):
        key = [key]

    results = OrderedDict([(k, list()) for k in key])

    for i in files:
        path_ = os.path.join(path, i)
        f = torch.load(path_)

        for k in key:
            result = fn(_fetch(f, k.split(".")))
            results[k].append(result)

        if id is not None:
            results = OrderedDict([(x, y[0]) for x,y in results.items()])
            break

    if len(key) == 1:
        results = results[key[0]]

    return results

def replace(path_):

    path = lambda x: os.path.join(path_, x)
    if 'train.pth' not in os.listdir(path_):
        if 'experiment.pth' in os.listdir(path_):
            try:
                f = torch.load(path('experiment.pth'))
            except (RuntimeError, EOFError) as e:
                with open(path('experiment.pth'), 'rb') as F:
                    f = pickle.load(F)

            if 'meta.pth' not in os.listdir(path_):
                with open(path('meta.pth'), 'wb') as F:
                    pickle.dump(f['meta'], F)

            if 'train.pth' not in os.listdir(path_):
                with open(path('train.pth'), 'wb') as F:
                    pickle.dump(f['train'], F)

            if 'test.pth' not in os.listdir(path_):
                with open(path('test.pth'), 'wb') as F:
                    pickle.dump(f['test'], F)

            if 'weights.pth' not in os.listdir(path_):
                with open(path('weights.pth'), 'wb') as F:
                    pickle.dump(f['weights'], F)

        else:

            with open(path('meta.pth'), 'wb') as F:
                pickle.dump(fetch_values(path_, 'meta', id=0), F)

            with open(path('train.pth'), 'wb') as F:
                value = denest(fetch_values(path_, 'train'))
                pickle.dump(value, F)

            with open(path('test.pth'), 'wb') as F:
                value = denest(fetch_values(path_, 'test'))
                pickle.dump(value, F)

            with open(path('weights.pth'), 'wb') as F:
                value = fetch_values(path_, 'weights')
                pickle.dump(value, F)


    if 'weights.pth' not in os.listdir(path_):
        try:
            f = torch.load(path('meta.pth'))
            with open(path('meta.pth'), 'wb') as F:
                pickle.dump(f, F)
        except (RuntimeError, pickle.UnpicklingError):
            pass

        try:
            f = torch.load(path('train.pth'))
            with open(path('train.pth'), 'wb') as F:
                pickle.dump(f, F)
        except (RuntimeError, pickle.UnpicklingError):
            pass

        try:
            f = torch.load(path('test.pth'))
            with open(path('test.pth'), 'wb') as F:
                pickle.dump(f, F)
        except (RuntimeError, pickle.UnpicklingError):
            pass

        try:
            f = torch.load(path('weights.pth'))
            with open(path('weights.pth'), 'wb') as F:
                pickle.dump(f, F)
        except (RuntimeError, pickle.UnpicklingError):
            pass
