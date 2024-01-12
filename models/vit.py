'''In this module, there are 2 classes of vision transformers. One is used to encode the training images and target images. The other is used as the predictor.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import einops

from .patch_embedding import PatchEmbedding
from .positional_encoder import PositionalEncoder
from .transformer_encoder import TransformerEncoderNetwork
from utils import apply_masks_over_embedded_patches


class VisionTransformerForPredictor(nn.Module):

    def __init__(self, num_patches, embedding_dim, predictor_embed_dim, feedforward_projection_dim, depth, num_heads, device, init_std=0.02, **kwargs):
        '''Vision Transformer to be used as the predictor. The output from the predictor will be in the same dimension as the input since the output is trying to predict the embedding of the target images.
        '''

        super(VisionTransformerForPredictor, self).__init__()

        self.predictor_embed = nn.Linear(embedding_dim, predictor_embed_dim).to(device) #to project the incoming embedding to the predictor's embedding dimension.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)).to(device) #learnable token for the masks.
        self.device = device
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        self.transformer_blocks = TransformerEncoderNetwork(
                                                            transformer_network_depth=depth,
                                                            device=self.device,
                                                            input_dim=self.embedding_dim, 
                                                            feedforward_projection_dim=feedforward_projection_dim,
                                                            num_heads=num_heads,
                                                            **kwargs
                                                            ).to(device)
        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(embedding_dim).to(device)

        self.predictor_projector = nn.Linear(predictor_embed_dim, embedding_dim).to(device) 


    def forward(self, x, masks_ctxt, masks_pred_target):

        x = self.predictor_embed(x)

        batch_size = len(x) // len(masks_ctxt) #this has to be done because the  apply_masks_over_embedded_patches function from utils.py causes the input tensor to be expanded by the number of masks given. In other words, the batch dimension value will be (original batch num x number of masks). Therefore, to get the right batch number, we do the division. REMEMBER, the function mentioned before is used with the context mask before the output comes to the predictor.

        #generate the positional embedding tokens using the original image embedding sizes. Not the tensor after any mask(s) applied. 
        #the positional embeddings have to be masked with both the masks (pred/target masks and context masks) separately.
        pos_embed_module = PositionalEncoder(token_length=self.num_patches, output_dim=self.embedding_dim, n=10000, device=self.device)
        pos_embedding_pred_target = pos_embed_module() #for the pred/target mask 
        
        #repeat the positional embedding's first dimension to match the Batch dimension.
        pos_embedding_pred_target = einops.repeat(pos_embedding_pred_target.unsqueeze(0), '() p e -> b p e', b=batch_size)

        #clone the same positional embedding tensor for the context mask.
        pos_embedding_ctxt = pos_embedding_pred_target.clone().detach() #for the context mask

        
        #apply the masks on the appropriate positional embedding. 
        pos_embedding_ctxt = apply_masks_over_embedded_patches(pos_embedding_ctxt, masks_ctxt)
        pos_embedding_pred_target = apply_masks_over_embedded_patches(pos_embedding_pred_target, masks_pred_target)  

        #we basically took the image size embedding, mask it to the context and added it to the context-masked image embedding that came from the encoder.
        x += pos_embedding_ctxt
        
        #we will be adding the pred tokens to the positional embedding that's been masked by the pred/target masks.
        pred_tokens = self.mask_token.repeat(pos_embedding_pred_target.size(0), pos_embedding_pred_target.size(1), 1) 
        pred_tokens += pos_embedding_pred_target






        return None        



        


class VisionTransformerForEncoder(nn.Module):

    def __init__(self, image_size, patch_size, in_channel, embedding_dim, depth, num_heads, device, init_std=0.02, **kwargs):
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
        self.init_std = init_std
        self.device = device

        self.patch_embed = PatchEmbedding(patch_size=self.patch_size, in_channel=self.in_channel, embedding_dim=self.embedding_dim, device=self.device).to(device)


        self.transformer_blocks = TransformerEncoderNetwork(
                                                            transformer_network_depth=depth,
                                                            device=device,
                                                            input_dim=embedding_dim, 
                                                            num_heads=num_heads,
                                                            **kwargs
                                                            ).to(device)
        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(self.embedding_dim).to(device)

                                                                

        
    
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
        
        
if __name__ == '__main__':


    device = torch.device('cuda:0')

    v = VisionTransformerForPredictor(embedding_dim=512, predictor_embed_dim=512, projection_dim_keys=512, projection_dim_values=512, feedforward_projection_dim=512, depth=5, num_heads=8, attn_dropout_prob=0.1, feedforward_dropout_prob=0.1, device=device, init_std=0.02)
    
    x = torch.randn((2, 196, 512))

    ret = v(x)
    print(ret)







