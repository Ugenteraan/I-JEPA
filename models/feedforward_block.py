'''Implementation of a single feedforward block in a transformer encoder.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

class FeedForwardEncoderBlock(nn.Sequential):

    def __init__(self, patch_embedding_dim, feedforward_projection_dim, feedforward_dropout_prob):

        #let's define the sequence using the nn.sequential's init itself.
        super().__init__(
                nn.Linear(patch_embedding_dim, feedforward_projection_dim),
                nn.GeLU(),
                nn.Dropout(feedforward_dropout_prob),
                nn.Linear(feedforward_projection_dim, patch_embedding_dim)
                )

