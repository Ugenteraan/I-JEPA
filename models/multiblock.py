'''Mask Collator module that calculates the mask.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import math
from multiprocessing import Value


class MultiBlockMaskCollator(object):
    '''This module will be supplied in the collate parameter during the initialization of dataloader.
       KEEP IN MIND THAT the best value for patch_size is 14. Anything more than 14 will cause the context mask to not properly cover the sides of the images. There'll always be a padding at the sides. I believe this happens due to the randomization of the block masks. When the block masks are randomized with the given size ranges, it cannot cover the sides when the patch size are more than the number of patches in total. When we set the patch size to be 14, the number of patches are 16. This is the ideal setting for an image of 224 x 224. If the image size is bigger, however, then the patch size can be increased. Just make sure, the patch size is divisible by the image size. 
    '''


    def __init__(self, 
                 image_size=224, 
                 patch_size=14,
                 num_context_mask=1,
                 num_pred_target_mask=4,
                 context_mask_scale=(0.85, 1.0),
                 pred_target_mask_scale=(0.15, 0.2),
                 aspect_ratio=(0.75, 1.5),
                 min_mask_length=4,
                 allow_overlap=False,
                 logger=None):
        
        super(MultiBlockMaskCollator, self).__init__()
        self.patch_size = patch_size
        self.num_patch_rows = image_size // patch_size #number of patches in the row dimension
        self.num_patch_cols = image_size // patch_size #number of patches in the col dimension
        self.context_mask_scale = context_mask_scale
        self.pred_target_mask_scale = pred_target_mask_scale
        self.aspect_ratio = aspect_ratio
        self.num_context_mask = num_context_mask
        self.num_pred_target_mask = num_pred_target_mask
        self.min_mask_length = min_mask_length
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)
        self.logger = logger
    
    def seed_step(self):
        '''This module will ensure that the value of the seed increases consistently across ALL workers.
           In my understanding, when multiple workers are used in the dataloader, each worker will be initialized with different random numbers. This might produce inconsistent behaviours.
           Therefore, to remedy this, we are using the seed here to ensure that the value of the generated number is consistent across the workers for a particular iteration. 
           However, when I tested the script with AND without the seed step, there were no noticeable inconsistencies. Perhaps it's due to the lower number of batch and data limit that I used for the testing. Regardless, this function seems to be safe to be around.
        '''
        i = self._itr_counter
        
        #In my understanding, using get_lock() will ensure that the operations performed AFTER it is done on the shared resource i.e. on all woekers. 
        with i.get_lock():
            i.value +=1
            v = i.value

        return v

    def randomize_block_size(self, scale, aspect_ratio, torch_generator=None):
        '''Given the scale and aspect ratio, we randomly generte a height and width for the mask blocks.
           The scale is responsible for determining how small/big the mask is going to be while the aspect ratio is responsible for the height and width ratio of the mask block.

        '''

        _rand = torch.rand(1, generator=torch_generator).item() 
        
        #here, we are randomizing the mask's scale. In other word, how many % we want to up/down scale the mask size in its entirety.
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s) #this formula will ensure the scale is within the given min max of the scale parameter.
        max_keep = int(self.num_patch_rows * self.num_patch_cols * mask_scale) #this calculation is needed to determine the height and width later.
        
        min_asp_ratio, max_asp_ratio = aspect_ratio
        block_aspect_ratio = min_asp_ratio + _rand * (max_asp_ratio - min_asp_ratio) #again this formula will ensure that the block's aspect ratio is within the parameter range.
        
        #in order to get different values for h and w, we use the multiplier and divisor operator in each one. It doesn't matter which. Since we're square rooting it, the value will be closeby.
        h = int(math.sqrt(max_keep * block_aspect_ratio))
        w = int(math.sqrt(max_keep / block_aspect_ratio))
        
        #if in any case the height and weight of the block exceeds the number of patches (in either dimension), the h and w will be reduced.
        while h >= self.num_patch_rows:
            h -= 1
        while w >= self.num_patch_cols:
            w -= 1

        return h,w
    
    def constrain_mask_to_acceptable_regions(self, mask, acceptable_regions=None, minus_num_region=0):
        '''The minus_num_region parameter will ensure that if the mask cannot be found in the acceptable regions, then we will reduce the number of acceptable regions (note that the acceptable regions contains more than one region. Therefore, we're reducing the number of the regions in the list). 
        '''

        num_of_regions = max(int(len(acceptable_regions) - minus_num_region), 0)

        for i in range(num_of_regions):
            mask *= acceptable_regions[i] #this code will ensure only the part of the regions are accepted in the mask. Whichever region with a value of 0 will be zero-ed in the mask as well.

        return mask

    def get_block_mask(self, block_size, acceptable_regions=None, num_tries=20, mask_complement_required=True):
        '''Get a single block mask based on the block size and the given acceptable region (from the original patch image) if given.
        '''

        h, w = block_size

        valid_mask = False
        minus_num_region = 0

        tries = num_tries #number of tries to find the valid mask. If it's unsuccessful, the acceptable_regions will be decreased.

        while not valid_mask:

            
            #self.num_patch_rows/width - h/w here is to find the "free" spots in the entire patch size.
            top = torch.randint(0, self.num_patch_rows - h, (1,))
            left = torch.randint(0, self.num_patch_cols - w, (1,))


            mask = torch.zeros((self.num_patch_rows, self.num_patch_cols), dtype=torch.int32)

            mask[top:top+h, left:left+w] = 1

            #if the acceptable regions are provided, then the mask should only be applied to those regions. Anything outside the regions should be left 0.
            if acceptable_regions is not None:
                mask = self.constrain_mask_to_acceptable_regions(mask=mask,
                                                                 acceptable_regions=acceptable_regions,
                                                                 minus_num_region=minus_num_region)
            #torch.nonzero returns a tensor containing the indices of all non-zero elements of input. 
            mask = torch.nonzero(mask.flatten()) #we are interested in only the non-zero indices. Not the entire num_patch_rows x num_patch_cols tensor.

            #since the mask might have been severely constrained if the acceptable regions were given, we need to make sure that the masks are biggeer than the minimum mask area we defined.
            valid_mask = len(mask) > self.min_mask_length

            if not valid_mask:
                tries -= 1
                
                if tries == 0:

                    minus_num_region += 1
                    tries = num_tries

                    self.logger.error(f"Valid mask not found, decreasing the number of acceptable regions [{minus_num_region}]")
                    
                
        mask = mask.squeeze()
        
        if mask_complement_required:
            #if we're generating masks for the target/predictor network, then we're gonna have to return the mask complements as well for it to be used for the context mask generator later.
            #basically, we're doing the same thing as before except reversed. But keep in mind that this is a full mask (complement) not its indices.
            mask_complement = torch.ones((self.num_patch_rows, self.num_patch_cols), dtype=torch.int32)
            mask_complement[top:top+h, left:left+w] = 0 


            return mask, mask_complement
        
        #if mask_complement not required.
        return mask



    def __call__(self, batch_data):
        '''Find #num_context_mask context mask(s) and #num_pred_target_mask prediction/target mask. 
        1) We first need to find the prediction/target masks. 
        2) Using the complement of the masks from #1, we find the context mask.
        '''
        
        
        num_batch = len(batch_data)

        collated_batch_data_images = torch.utils.data.default_collate([x['images'] for x in batch_data]) #we return the original data here since masking processes does not require the data.

        #before we randomize anything, we need to remember that if multiple workers are accessing this module, then the random values would be different for the masks on the same image!
        #hence, we have to use the torch generator that uses a same seed value across the workers for the same image.
        seed = self.seed_step()

        torch_generator = torch.Generator()
        torch_generator.manual_seed(seed) #feed the manual seed to the generator.

        
        #we want the prediction/target masks to have randomized height and width in every iteration.
        pred_target_mask_size = self.randomize_block_size(scale=self.pred_target_mask_scale, aspect_ratio=self.aspect_ratio, torch_generator=torch_generator) 
        context_mask_size = self.randomize_block_size(scale=self.context_mask_scale, aspect_ratio=(1.,1.), torch_generator=torch_generator) #we maintain the 1 to 1 aspect ratio for context mask blocks. 

        #these variables are used to make sure the length of all the masks are the same so that the collate function works.
        #REMEMBER, since we're randomizing the masks size (or length when flattened), there's bound to be inconsistencies.
        min_keep_pred_target, min_keep_context = self.num_patch_rows*self.num_patch_cols, self.num_patch_rows*self.num_patch_cols

        batch_masks_pred_target, batch_masks_context = [], []
        for _ in range(num_batch):

            array_masks_pred_target, array_masks_complement = [], []

            for idxmask in range(self.num_pred_target_mask):
                mask_pred_target, mask_complement = self.get_block_mask(pred_target_mask_size)
                array_masks_pred_target.append(mask_pred_target)
                array_masks_complement.append(mask_complement)
                min_keep_pred = min(min_keep_pred_target, len(mask_pred_target))

            batch_masks_pred_target.append(array_masks_pred_target) #append all the masks for this element in the batch.

            acceptable_regions = array_masks_complement
            if self.allow_overlap:
                acceptable_regions = None

            array_masks_context = []
            for _ in range(self.num_context_mask):

                #the idea here is to use a bigger mask scale than the predcit/target mask and constrain it to the acceptable regions. 
                mask_context = self.get_block_mask(context_mask_size, acceptable_regions=acceptable_regions, mask_complement_required=False)
                array_masks_context.append(mask_context)

                min_keep_context = min(min_keep_context, len(mask_context))

            batch_masks_context.append(array_masks_context)
        
        
        collated_masks_pred_target = [[m[:min_keep_pred] for m in mpt] for mpt in batch_masks_pred_target]
        collated_masks_pred_target = torch.utils.data.default_collate(collated_masks_pred_target)

        collated_masks_context = [[m[:min_keep_context] for m in mc] for mc in batch_masks_context]
        collated_masks_context = torch.utils.data.default_collate(collated_masks_context)
        
        #returns the data in [number of total masks, batch size, mask]. Later when we iterate through the masks, we'll get all the n-th masks for all the batches
        return {
                'collated_batch_data_images':collated_batch_data_images,
                'collated_masks_pred_target':collated_masks_pred_target,
                'collated_masks_context': collated_masks_context
                }

        



