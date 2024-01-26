'''Transformer Encoder blocks that contain multi-head attention modules and feed forward encoders. 
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from .mhsa import MultiHeadAttention
from .feedforward_block import FeedForwardEncoderBlock

class TransformerEncoderBlock(nn.Module):
    '''A single transformer encoder block.
    '''


    def __init__(self, input_dim, projection_keys_dim, projection_values_dim, num_heads, attn_dropout_prob, feedforward_projection_dim, feedforward_dropout_prob, device):

        super(TransformerEncoderBlock, self).__init__()

        #initialize the self attention module together with a layernorm layer.
        self.multi_head_attention_block = nn.Sequential(
                                                        nn.LayerNorm(input_dim),
                                                        MultiHeadAttention(input_dim=input_dim,
                                                                           projection_keys_dim=projection_keys_dim,
                                                                           projection_values_dim=projection_values_dim,
                                                                           num_heads=num_heads,
                                                                           attn_dropout_prob=attn_dropout_prob).to(device)
                                                        )

        #initialize the feedforward block together with a layernorm layer.
        self.feedforward_block = nn.Sequential(
                                                nn.LayerNorm(input_dim),
                                                FeedForwardEncoderBlock(input_dim=input_dim,
                                                                        feedforward_projection_dim=feedforward_projection_dim,
                                                                        feedforward_dropout_prob=feedforward_dropout_prob).to(device)
                                                )
        

    def forward(self, x):

        multihead_attention_output = self.multi_head_attention_block(x)
        multihead_attention_output += x #skip connection 

        feedforward_output = self.feedforward_block(multihead_attention_output)
        feedforward_output += multihead_attention_output

        return feedforward_output




class TransformerEncoderNetwork(nn.Sequential):
    '''Created a network of transformers using the TransformerEncoderBlock module with the given depth.
    '''
    
    def __init__(self, transformer_network_depth, input_dim, device, **kwargs):

        super().__init__(*[TransformerEncoderBlock(input_dim=input_dim,
                                                   device=device,
                                                   **kwargs
                                                   ) for _ in range(transformer_network_depth)])










