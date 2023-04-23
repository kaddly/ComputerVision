import torch
from torch import nn
from module.ViTBlock import *


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
