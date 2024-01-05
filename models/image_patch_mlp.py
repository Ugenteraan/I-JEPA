'''Module to produce linear projection of the patched images (tokens).
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn



class PatchEmbedding(nn.Module):
    '''The idea of patch embedding is to divide a given image into same-size grids and project each grids into another dimension linearly.
       This can be done effectively by using a conv layer with specific configuration.
    '''


    def __init__(self, image_height, image_width, patch_size, in_channel, embedding_dim):

        super(PatchEmbedding, self).__init__()

        self.patch_projection = nn.Conv2d(in_channel, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        
        x = self.patch_projection(x) #the output of this will be [batch size, embedding_dim, num_patches in height dimension, num_patches in width dimension]
        x = x.flatten(2) #flatten both the num_patches dimension to get the total number of patches. [batch size, embedding_dim, num_patches]
        x = x.transpose(1,2) #swap the axis. [batch size, num_patches, embedding_dim]

        return x



'''
if __name__ == '__main__':

    p = PatchEmbedding(image_height=224, image_width=224, patch_size=16, in_channel=3, embedding_dim=256)
    
    img = torch.randn((1, 3, 224, 224))

    x = p(img)
    print(x.shape)
'''
