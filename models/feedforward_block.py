'''Implementation of a single feedforward block in a transformer encoder.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

class FeedForwardEncoderBlock(nn.Sequential):

    def __init__(self, input_dim, mlp_ratio, mlp_dropout_prob):

        #let's define the sequence using the nn.sequential's init itself.
        super().__init__(
                nn.Linear(input_dim, input_dim*mlp_ratio),
                nn.GELU(),
                nn.Dropout(mlp_dropout_prob),
                nn.Linear(input_dim*mlp_ratio, input_dim)
                )

