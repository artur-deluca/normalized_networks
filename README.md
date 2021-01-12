# Normalized networks
Set of experiments inspired by Poggio et al. 2020 - Complexity control by gradient descent in deep networks

```python
> python train --help
Run experiments to identify weight dynamics

optional arguments:
  -h, --help           show this help message and exit
  -l , --layers        Intermediate layers (default: 1)
  -u , --units         Intermediate units (default: 10)
  -e , --epochs        Training epochs (default: 200)
  --lr                 Learning rate (default: 0.01)
  -s , --save          Directory to store the experiments (default:
                       ./experiments/)
  -rs , --seed         Random seed (default: 34)
  -a , --alpha         Regularization factor (default: 0.0)
  -bs , --batch_size   Batch size (default: 0)
  --quiet              Verbosity mode (default: True)
  --parametric         Parametric mode (default: False)
  --normalize          Normalize (default: False)
```
