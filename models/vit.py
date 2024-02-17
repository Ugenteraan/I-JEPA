'''In this module, there are 2 classes of vision transformers. One is used to encode the training images and target images. The other is used as the predictor.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch
import einops
import einops.layers.torch as einops_torch

from .patch_embedding import PatchEmbedding
from .positional_encoder import PositionalEncoder
from .transformer_encoder import TransformerEncoderNetwork
from utils import apply_masks_over_embedded_patches


class VisionTransformerForPredictor(nn.Module):

    def __init__(self, input_dim, num_patches, predictor_network_embedding_dim, device, num_class, classification_embedding_dim, **kwargs):
        '''Vision Transformer to be used as the predictor. The output from the predictor will be in the same dimension as the input since the output is trying to predict the embedding of the target images.
        '''

        super(VisionTransformerForPredictor, self).__init__()

        self.predictor_embed = nn.Linear(input_dim, predictor_network_embedding_dim).to(device) #to project the incoming embedding to the predictor's embedding dimension.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_network_embedding_dim)).to(device) #learnable token for the masks.
        self.device = device
        self.predictor_network_embedding_dim = predictor_network_embedding_dim
        self.num_patches = num_patches #we have to define the num patches here since we can't infer this from the input. Reason: Input is already masked. So, the num of patches is lower than it should be.
        self.transformer_blocks = TransformerEncoderNetwork(device=device,
                                                              input_dim=self.predictor_network_embedding_dim, 
                                                              **kwargs
                                                              ).to(device)




        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(input_dim).to(device)

        # self.predictor_projector = nn.Linear(self.predictor_network_embedding_dim, input_dim).to(device) 

        self.classification_head = nn.Sequential(einops_torch.Reduce('b n e -> b e', reduction='mean'),
                                                 nn.LayerNorm(self.predictor_network_embedding_dim),
                                                 nn.Linear(self.predictor_network_embedding_dim, classification_embedding_dim),
                                                 nn.GELU(),
                                                 nn.Linear(classification_embedding_dim, num_class)).to(self.device)



    def forward(self, x, masks_ctxt=None, masks_pred_target=None):

        x = self.predictor_embed(x)

        batch_size = x.size(0) #this has to be done because the  apply_masks_over_embedded_patches function from utils.py causes the input tensor to be expanded by the number of masks given. In other words, the batch dimension value will be (original batch num x number of masks). Therefore, to get the right batch number, we do the division. REMEMBER, the function mentioned before is used with the context mask before the output comes to the predictor.
        
        # _, num_ctxt, _ = x.size() #to get the size of the context mask.

        #generate the positional embedding tokens using the original image embedding sizes. Not the tensor after any mask(s) applied. 
        #the positional embeddings have to be masked with both the masks (pred/target masks and context masks) separately.
        pos_embed_module = PositionalEncoder(token_length=self.num_patches, output_dim=self.predictor_network_embedding_dim, n=10000, device=self.device)
        pos_embedding_pred_target = pos_embed_module() #for the pred/target mask 
        
        #repeat the positional embedding's first dimension to match the Batch dimension.
        pos_embedding_pred_target = einops.repeat(pos_embedding_pred_target.unsqueeze(0), '() p e -> b p e', b=batch_size)

        #clone the same positional embedding tensor for the context mask.
        pos_embedding_ctxt = pos_embedding_pred_target.clone().detach() #for the context mask

        
        # #apply the masks on the appropriate positional embedding. 
        # pos_embedding_ctxt = apply_masks_over_embedded_patches(pos_embedding_ctxt, masks_ctxt)
        # pos_embedding_pred_target = apply_masks_over_embedded_patches(pos_embedding_pred_target, masks_pred_target)  

        #we basically took the image size embedding, mask it to the context and added it to the context-masked image embedding that came from the encoder.
        x += pos_embedding_ctxt
        
        # #we will be adding the pred tokens to the positional embedding that's been masked by the pred/target masks.
        # pred_tokens = self.mask_token.repeat(pos_embedding_pred_target.size(0), pos_embedding_pred_target.size(1), 1) 
        # pred_tokens += pos_embedding_pred_target

        # #next, the pred tokens has to be concatenated to the x. The tokens are to be concatenated in the first dimension. REMEBER, in the first dimension of 'x', where the patch index should be, all the pred/target masks indices shouldn't be present since the context mask is a complement of them. The idea here is to concat the indices to 'x' and eventually, the network will learn to predict the embeddings in those pred/target indices. 
        # x = x.repeat(len(masks_pred_target), 1, 1) #we need to repeat the batch dimension first in order to concat with the pred_tokens.
        # x = torch.cat([x, pred_tokens], dim=1)

        #run through the predictor network.
        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)

        #after the input goes through the transformer network, we want to return the predictions to perform target loss. However, we do not need the context mask. Since we concatenated everything along the way, the first 'num_ctxt' value in the first dimension, can be ignored. 
        # x = x[:, num_ctxt:]
        x = self.classification_head(x)

        return x



class VisionTransformerForEncoder(nn.Module):

    def __init__(self, image_size, patch_size, image_depth, encoder_network_embedding_dim, device, num_classes, **kwargs):
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

        
        # self.classification_head = nn.Sequential(einops_torch.Reduce('b n e -> b e', reduction='mean'),
        #                                          nn.LayerNorm(self.encoder_network_embedding_dim),
        #                                          nn.Linear(self.encoder_network_embedding_dim, self.encoder_network_embedding_dim*2),
        #                                          nn.GELU(),
        #                                          nn.Linear(self.encoder_network_embedding_dim*2, num_classes))

        #apply layernorm on the output of the transformer blocks.
        self.final_layernorm = nn.LayerNorm(self.encoder_network_embedding_dim).to(device)

        
    
    def forward(self, x, masks=None):
        '''x: Torch image [batch size, num_channels, image_size, image_size]
           return: embedding of the masked image.
        '''

        #patch embed the images.
        x = self.patch_embed(x)

        #generate the positional embedding tokens
        pos_embed_module = PositionalEncoder(token_length=x.size(1), output_dim=x.size(2), n=10000, device=self.device)
        pos_embedding_tensor = pos_embed_module()
        

        stacked_pos_enc_tensor = einops.repeat(pos_embedding_tensor.unsqueeze(0), '() p e -> b p e', b=x.size(0)).detach()
        #concat the pos embedding tensor the patch embedding.
        x = x + stacked_pos_enc_tensor

        if masks is not None:
            x = apply_masks_over_embedded_patches(x, masks)

        x = self.transformer_blocks(x)
        x = self.final_layernorm(x)

        return x
        
'''        
if __name__ == '__main__':


    device = torch.device('cuda:0')

    v = VisionTransformerForPredictor(encoder_network_embedding_dim=512, predictor_network_embed_dim=512, projection_dim_keys=512, projection_dim_values=512, feedforward_projection_dim=512, depth=5, num_heads=8, attn_dropout_prob=0.1, feedforward_dropout_prob=0.1, device=device, init_std=0.02)
    
    x = torch.randn((2, 196, 512))

    ret = v(x)
    print(ret)
'''



