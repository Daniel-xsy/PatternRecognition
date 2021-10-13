import torch
import torch.nn as nn

def init_weight(model):
    for layer in model.modules():
        if type(layer) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_normal_(layer.weight)
    