# -*- coding: utf-8 -*-
import pickle
import torch
import argparse

import time
import numpy as np
import json

from pn import PolicyNetwork
import copy

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        m.bias.data.fill_(0)
        
PN_model = PolicyNetwork(input_dim=29, dim1=64, output_dim=12)
PN_model.apply(weights_init_normal)
torch.save(PN_model.state_dict(), '../PN-model-ours/PN-model-ours.txt')
torch.save(PN_model.state_dict(), '../PN-model-ours/pretrain-model.pt')
