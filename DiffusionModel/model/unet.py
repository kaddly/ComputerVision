import torch
from torch import nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super.__init__()
        self.beta = beta
        self.step = 0
