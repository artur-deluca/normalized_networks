import tempfile
import inspect
import os
from collections import namedtuple, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optimizers

try:
    from utils import load
except ImportError:
    from .utils import load

pair = namedtuple("pair", "train, test")
_data = namedtuple("data", "X, y")
results = namedtuple("results", "meta, accuracy, loss, gradient, weights")
meta = namedtuple("meta", "layers, units, lr, optim")

def loss_fn(net, alpha):
    def loss(output, target):
        loss = torch.mean(torch.exp(-(output*target)))
        if alpha:
            reg_loss = 0
            for k, v in net.named_parameters():
                if not(k.startswith('scalar')):
                    reg_loss += (torch.norm(v) - 1)**2
            loss += alpha * reg_loss
        return loss

    return loss

def get_scalar(param):
    if param:
        return nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    return 1

class Model(nn.Module):
    def __init__(self, param, n_layers=2, lr=0.05, optim="Adam", units=10, alpha=0, **kwargs):
        super(Model, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(3, units, bias=False)])
        self.scalar0 = get_scalar(param)
        self.scalars = [self.scalar0]

        for i in range(n_layers):
            setattr(self, f'scalar{i+1}', get_scalar(param))
            self.scalars.append(getattr(self, f'scalar{i+1}'))
            self.layers.append(nn.Linear(units, units, bias=False))

        setattr(self, f'scalar{i+2}', get_scalar(param))
        self.scalars.append(getattr(self, f'scalar{i+2}'))
        self.layers.append(nn.Linear(units, 1, bias=False))

        self.criterion = loss_fn(self, alpha)

        optimizer = optimizers.__dict__[optim]
        args = {k: v for k, v in kwargs.items() if k in inspect.getargspec(optimizer).args}
        self.optim = optimizer(self.parameters(), lr=lr, **args)
        self._initialize_weights(kwargs.get('seed', None))
        self._meta = meta(n_layers, units, lr, optim)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(self.scalars[i]*layer(x))
        x = (self.scalars[-1]*self.layers[-1](x)).view(-1)
        #x = torch.tanh(self.scalars[-1]*self.layers[-1](x)).view(-1)
        return x

    def normalize(self):
        m = dict(self.named_parameters())
        for k, v in m.items():
            if not k.startswith('scalar'):
                m[k] = v / v.norm()
        self.load_state_dict(m)

    def _initialize_weights(self, seed):
        if seed: torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)

            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        self.normalize()

    def _collect_activations(self, x):

        activations = dict()
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(self.scalars[i]*layer(x))
            activations[i] = x.cpu().data.numpy()
        x = (self.scalars[-1]*self.layers[-1](x)).view(-1)
        #x = torch.tanh(self.scalars[-1]*self.layers[-1](x)).view(-1)
        return x, activations

    def fit(self, data, epochs: int, batch_size: int = 0, norm: bool = False, verbose: bool = True, path: str = None, stop: bool = False):

        _, filename = tempfile.mkstemp()
        if not batch_size:
            batch_size = int(len(data.train.X))
            batches = 1
        else:
            batches = len(data.train.X) // batch_size

        for epoch in range(epochs + 1):

            self.optim.zero_grad()

            torch.save(self.state_dict(), filename)
            history = {"meta": self._meta._asdict(), "weights": torch.load(filename), "epoch": epoch}
            history["test"] = self._process_data(data.test, trainable=False)
            history["train"] = self._process_data(data.train, trainable=False)

            for i in range(int(batches)):
                batch = slice(i*batch_size, (i+1)*batch_size)
                batch =  _data(data.train.X[batch], data.train.y[batch])
                _ = self._process_data(batch, trainable=True)

            if norm: self.normalize()

            if path:
                os.makedirs(path, exist_ok=True)
                torch.save(history, os.path.join(path, f"{epoch}.pth"))

            if verbose:
                print(
                    (f'Epoch: {epoch}/{epochs} | '
                     f'Loss: {history["train"]["loss"]:.3f} | '
                     f'Accuracy: {history["train"]["accuracy"]:.3f}'),
                    end="\r",
                )

            if stop and history["train"]["accuracy"] == 1.000:
                break

        if verbose:
            print()

    def _process_data(self, data, trainable=False):

        outputs, activations = self._collect_activations(data.X)
        loss = self.criterion(outputs, data.y)

        results = {
            "loss": loss.item(),
            "accuracy": accuracy(outputs, data.y),
            "gradient": gradient(self, loss),
            "activations": activations,
        }

        if trainable:
            loss.backward()
            self.optim.step()

        return results

def gradient(model, loss):
    grad = flatten(autograd.grad(loss, model.parameters(), create_graph=True))
    return grad.cpu().data.numpy()

def from_logs(path, epoch):

    meta = load(path, 'meta')
    weights = load(path, 'weights')
    param = not('non_param' in path)

    layers = meta['layers']
    units = meta['units']
    lr = meta['lr']
    optim = meta['optim']

    model = Model(param, layers, lr, optim, units)
    weights = weights[epoch]
    model.load_state_dict(weights)

    return model


def accuracy(outputs, labels):
    with torch.no_grad():
        predictions = torch.sign(outputs)

        correct = (predictions == labels).float().sum().item()

    return correct / len(labels)


def flatten(tensor):
    vector = torch.Tensor().contiguous()
    for g in tensor:
        vector = torch.cat([vector, g.contiguous().view(-1)])
    return vector

