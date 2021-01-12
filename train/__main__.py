import os
import numpy as np
import torch
from datetime import datetime

import model
import data
import replace
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(
    description="Run experiments to identify weight dynamics",
    formatter_class=ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-l", "--layers", metavar="", default=1, type=int, help="Intermediate layers"
)
parser.add_argument(
    "-u", "--units", metavar="", default=10, type=int, help="Intermediate units"
)
parser.add_argument(
    "-e", "--epochs", metavar="", default=200, type=int, help="Training epochs"
)
parser.add_argument(
    "--lr", metavar="", default=0.01, type=float, help="Learning rate"
)
parser.add_argument(
    "-s",
    "--save",
    metavar="",
    default="./experiments/",
    type=str,
    help="Directory to store the experiments",
)
parser.add_argument(
    "-rs", "--seed", metavar="", default=34, type=int, help="Random seed"
)
parser.add_argument(
    "-a", "--alpha", metavar="", default=.0, type=float, help="Regularization factor"
)
parser.add_argument(
    "-bs", "--batch_size", metavar="", default=0, type=int, help="Batch size"
)
parser.add_argument("--quiet", action="store_false", help="Verbosity mode")
parser.add_argument("--parametric", action="store_true", help="Parametric mode")
parser.add_argument("--normalize", action="store_true", help="Normalize")
parser.set_defaults(quiet=True)
parser.set_defaults(normalize=False)
parser.set_defaults(parametric=False)

args = parser.parse_args()
data = data.get_dataset(seed=args.seed)
path = 'parametric' if args.parametric else 'non_parametric'
path += f'_alpha_{str(args.alpha).replace(".", "")}'
if args.normalize:
    path += f'_normalize'

path += f'_{datetime.now().strftime("%m_%d_%Y_%I_%M")}'
path = os.path.join(args.save, path)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

i = 0
while True:
    if os.path.exists(path):
        path += str(i)
    else:
        break
    i+=1

net = model.Model(args.parametric, n_layers=args.layers, units=args.units, lr=args.lr, alpha=args.alpha, seed=args.seed, path=path)
net.fit(data=data, epochs=args.epochs, batch_size=args.batch_size, norm=args.normalize, verbose=args.quiet, path=path)
replace.replace(path)
for x in os.listdir(path):
    if x.strip('.pth').isnumeric():
        os.remove(os.path.join(path, x))
