'''This module is an exact replication of vit.py with the exception of having no mask token. DownstreamPredictor module at the end is where we'll be replacing the head of the predictor based on the use case of the downstreaming.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import einops.layers.torch as einops_torch

from .patch_embedding import PatchEmbedding
from .positional_encoder import PositionalEncoder
from .transformer_encoder import TransformerEncoderNetwork



class VisionTransformerForPredictor(nn.Module):

    def __init__(self, input_dim, predictor_network_embedding_dim, device, **kwargs):
        '''Vision Transformer to be used as the predictor. The output from the predictor will be in the same dimension as the input since the output is trying to predict the embedding of the target images.
        '''

        super(VisionTransformerForPredictor, self).__init__()

        self.predictor_embed = nn.Linear(input_dim, predictor_network_embedding_dim).to(device) #to project the incoming embedding to the predictor's embedding dimension.
        self.device = device
        self.predictor_network_embedding_dim = predictor_network_embedding_dim
        self.transformer_blocks = TransformerEncoderNetwork(device=device,
                                                              input_dim=self.predictor_network_embedding_dim, 
                                                              **kwargs
                                                              ).to(device)




        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(input_dim).to(device)

        #we will be removing this later.
        self.predictor_projector = nn.Linear(self.predictor_network_embedding_dim, input_dim).to(device) 



    def forward(self, x):
        '''Note that there is no more mask related logics and mask input to this function.
        '''
        
        x = self.predictor_embed(x)


        #run through the predictor network.
        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)

        x = self.predictor_projector(x) #still included here until the load completes.

        return x



class VisionTransformerForEncoder(nn.Module):

    def __init__(self, image_size, patch_size, image_depth, encoder_network_embedding_dim, device, **kwargs):
        '''Vision Transformer to be used as the encoder for both the training and target images. There is no CLS token since there's no classification going on here.
           MLP head is also not necessary for this ViT since we only want to produce embeddings.
        '''

        super(VisionTransformerForEncoder, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.image_depth = image_depth
        self.patch_embedding_dim = self.image_depth*self.patch_size**2
        self.encoder_network_embedding_dim = encoder_network_embedding_dim
        self.device = device

        self.patch_embed = PatchEmbedding(patch_size=self.patch_size, image_depth=self.image_depth, embedding_dim=self.encoder_network_embedding_dim, device=self.device).to(device)

        #use the nn.sequential module to build the transformer network with the specified depth.
        self.transformer_blocks = TransformerEncoderNetwork(device=device,
                                                            input_dim=self.encoder_network_embedding_dim,
                                                            **kwargs
                                                            ).to(device)

        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(self.encoder_network_embedding_dim).to(device)


        
    
    def forward(self, x):
        '''x: Torch image [batch size, num_channels, image_size, image_size]
           return: embedding of the image.
        '''

        #patch embed the images.
        x = self.patch_embed(x)

        #generate the positional embedding tokens
        pos_embed_module = PositionalEncoder(token_length=x.size(1), output_dim=x.size(2), n=10000, device=self.device)
        pos_embedding_tensor = pos_embed_module()
        
        #concat the pos embedding tensor the patch embedding.
        x = x + pos_embedding_tensor


        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)

        return x



class TrainedEncoder(nn.Module):
    '''In this module, we add positional embeddings to the encoder as they are not included in the saved .pth file.
    '''

    def __init__(self, trained_encoder, device='cpu', logger=None):

        super(TrainedEncoder, self).__init__()

        # #change the mode to eval first before anything.
        trained_encoder.eval()

        #we first have to separate the patch embedding from the rest of the trained encoder. The reason is to add the positional embedding tensors before the input is given to the rest of the transformer blocks.
        self.trained_patch_embed_encoder = list(trained_encoder.children())[0]
        self.trained_encoder_transformer_blocks = torch.nn.Sequential(*list(trained_encoder.children())[1:])

        self.device = device
        

        #disable gradient flow in both the trained networks.
        for param in self.trained_patch_embed_encoder.parameters():
            param.requires_grad = False

        for param in self.trained_encoder_transformer_blocks.parameters():
            param.requires_grad = False



   

    def forward(self, x):
        
        
        #--------Starting of the trained encoder process.
        x = self.trained_patch_embed_encoder(x)

        #generate the positional embedding tokens
        pos_embed_module = PositionalEncoder(token_length=x.size(1), output_dim=x.size(2), n=10000, device=self.device)
        pos_embedding_tensor = pos_embed_module()


        #concat the pos embedding tensor the patch embedding.
        x = x + pos_embedding_tensor

        self.trained_encoder_transformer_blocks.eval()
        x = self.trained_encoder_transformer_blocks(x)

        #--------Ending of the trained encoder process.

        ######################################################


        return x



class DownstreamHead(nn.Module):
    '''This module is the replacement for the head that was taken away in the predictor's network.
    '''

    def __init__(self, encoder_network_embedding_dim, classification_embedding_dim, num_class=77, device='cpu', logger=None, init_std=0.02):

        super(DownstreamHead, self).__init__()

        
        self.init_std = init_std
        self.device = device
        

        #new classification head to be trained.
        
        self.classification_head = nn.Sequential(einops_torch.Reduce('b n e -> b e', reduction='mean'),
                                                 nn.LayerNorm(encoder_network_embedding_dim),
                                                 nn.Linear(encoder_network_embedding_dim, classification_embedding_dim),
                                                 nn.GELU(),
                                                 nn.Linear(classification_embedding_dim, num_class)).to(self.device)

        self.apply(self._init_weights_)

    def _init_weights_(self, m):
        '''Weight init using truncated normal distribution: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        '''

        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):

        x = self.classification_head(x) #the newly added head.

        return x
