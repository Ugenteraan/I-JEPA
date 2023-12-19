'''Mask Collator module that calculates the mask.
'''

import torch
import math
from multiprocessing import Value


class MultiBlockMaskCollator:
    '''This module will be supplied in the collate parameter during the initialization of dataloader.
    '''


    def __init__(self, 
                 image_height=224, 
                 image_width=224, 
                 patch_size=14,
                 num_context_mask=1,
                 num_pred_target_mask=4,
                 context_mask_scale=(0.85, 1.0),
                 pred_target_mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.75, 1.5),
                 min_mask_length=4,
                 allow_overlap=False):
        
        self.patch_size = patch_size
        self.patch_height = image_height // patch_size
        self.patch_width = image_width // patch_size
        self.context_mask_scale = context_mask_scale
        self.pred_target_mask_scale = pred_target_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_context_mask = num_context_mask
        self.num_pred_target_mask = num_pred_target_mask
        self.min_mask_length = min_mask_length
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)
    
    def randomize_block_size(self, scale, aspect_ratio, torch_seed=None):
        '''Given the scale and aspect ratio, we randomly generte a height and width for the mask blocks.
           The scale is responsible for determining how small/big the mask is going to be while the aspect ratio is responsible for the height and width ratio of the mask block.

        '''

        _rand = torch.rand(1, generator=torch_seed).item() 
        
        #here, we are randomizing the mask's scale. In other word, how many % we want to up/down scale the mask size in its entirety.
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s) #this formula will ensure the scale is within the given min max of the scale parameter.
        max_keep = int(self.patch_height * self.patch_width * mask_scale) #this calculation is needed to determine the height and width later.
        
        min_asp_ratio, max_asp_ratio = aspect_ratio
        block_aspect_ratio = min_asp_ratio + _rand * (max_asp_ratio - min_asp_ratio) #again this formula will ensure that the block's aspect ratio is within the parameter range.
        
        #in order to get different values for h and w, we use the multiplier and divisor operator in each one. It doesn't matter which. Since we're square rooting it, the value will be closeby.
        h = int(math.sqrt(max_keep * block_aspect_ratio))
        w = int(math.sqrt(max_keep / block_aspect_ratio))
        
        #if in any case the height and weight of the block exceeds the number of patches (in either dimension), the h and w will be reduced.
        while h >= self.patch_height:
            h -= 1
        while w >= self.patch_width:
            w -= 1

        return h,w


    def __call__(self, batch_images):
        '''Find #num_context_mask context mask(s) and #num_pred_target_mask prediction/target mask. 
        1) We first need to find the prediction/target masks. 
        2) Using the complement of the masks from #1, we find the context mask.
        3) 
        '''

        num_batch = len(batch_images)
        
        #we want the prediction/target masks to have randomized height and width in every iteration.

        pred_target_mask_size = self.randomize_block_size(scale=self.pred_target_mask_scale, aspect_ratio=self.aspect_ratio) 
        context_mask_size = self.randomize_block_size(scale=self.context_mask_scale, aspect_ratio=(1.,1.)) #we maintain the 1 to 1 aspect ratio for context mask blocks. 

if __name__=='__main__':

    m = MultiBlockMaskCollator()
    print(m(batch_images=[1]))










