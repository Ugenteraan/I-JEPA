'''In this module, there are 2 classes of vision transformers. One is used to encode the training images and target images. The other is used as the predictor.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .patch_embedding import PatchEmbedding
from .positional_encoder import PositionalEncoder


class VisionTransformerForEncoder(nn.Module):

    def __init__(self, image_size, patch_size, in_channel, embedding_dim, depth, num_heads, attn_drop_rate, mlp_drop_rate, device, init_std=0.02, **kwargs):
        '''Vision Transformer to be used as the encoder for both the training and target images. There is no CLS token since there's no classification going on here.
        '''

        super(VisionTransformerForEncoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.mlp_drop_rate = mlp_drop_rate
        self.init_std = init_std
        self.device = device

        self.patch_embed = PatchEmbedding(patch_size=self.patch_size, in_channel=self.in_channel, embedding_dim=self.embedding_dim, device=self.device)

        
    
    def forward(self, x, masks=None):
        '''x: Torch image [batch size, num_channels, image_size, image_size]
           return: embedding of the masked image.
        '''

        #patch embed the images.
        x = self.patch_embed(x)

        pos_embed = PositionalEncoder(token_length=x.size(1), output_dim=x.size(2), n=10000, device=self.device)
        

