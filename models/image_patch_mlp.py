'''Module to produce linear projection of the patched images (tokens).
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn



class ImagePatchMLP(nn.Module):
    '''A single layer of fully-connected network to project the flattened image patches into a lower/higher dimensional space.
    '''


    def __init__(self, in_dim, out_dim):

        super(ImagePatchMLP, self).__init__()

        self.single_layer = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        '''Project the given image patches (flattened) without an activation function to another space.
            
          Input:
            A tensor of shape [batch size, total number of patches, a single flattened image dimension]
        '''
