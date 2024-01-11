'''In this module, there are 2 classes of vision transformers. One is used to encode the training images and target images. The other is used as the predictor.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch

from .patch_embedding import PatchEmbedding
from .positional_encoder import PositionalEncoder
from .transformer_encoder import TransformerEncoderNetwork
from utils import apply_masks_over_embedded_patches


class VisionTransformerForPredictor(nn.Module):

    def __init__(self, embedding_dim, predictor_embed_dim, depth, num_heads, attn_drop_rate, mlp_drop_rate, device, init_std=0.02, **kwargs):

        super(VisionTransformerForPredictor, self).__init__()

        self.predictor_embed = nn.Linear(embedding_dim, predictor_embed_dim) #to project the incoming embedding to the predictor's embedding dimension.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) #learnable token for the masks.

        


class VisionTransformerForEncoder(nn.Module):

    def __init__(self, image_size, patch_size, in_channel, embedding_dim, depth, num_heads, attn_drop_rate, mlp_drop_rate, device, init_std=0.02, **kwargs):
        '''Vision Transformer to be used as the encoder for both the training and target images. There is no CLS token since there's no classification going on here.
           MLP head is also not necessary for this ViT since we only want to produce embeddings.
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


        self.transformer_blocks = TransformerEncoderNetwork(patch_embedding=patch_embedding,
                                                            device=device,
                                                            patch_embedding_dim=patch_embedding_dim, 
                                                            projection_dim_keys=projection_dim_keys,
                                                            projection_dim_values=projection_dim_values, 
                                                            num_heads=num_heads
                                                            )
        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(self.embedding_dim)

                                                                

        
    
    def forward(self, x, masks=None):
        '''x: Torch image [batch size, num_channels, image_size, image_size]
           return: embedding of the masked image.
        '''

        #patch embed the images.
        x = self.patch_embed(x)

        #generate the positional embedding tokens
        pos_embed_module = PositionalEncoder(token_length=x.size(1), output_dim=x.size(2), n=10000, device=self.device)
        pos_embedding_tensor = pos_embed_module()
        
        #concat the pos embedding tensor the patch embedding.
        x = x + pos_embedding_tensor

        if masks is not None:
            x = apply_masks_over_embedded_patches(x, masks)

        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)

        return x
        
        

