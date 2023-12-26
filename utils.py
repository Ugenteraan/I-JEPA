'''Helper functions to be used throughout the project.
'''

import cv2
import torch
import numpy as np



def apply_masks_over_image_patches(image, patch_size, image_height, image_width, masks_array, negate_mask=True):
    '''Applies patched masks on the original image and returns the resulting image.
       negate_mask parameter is used to reverse the mask's purpose. That is to only show the masked areas instead of block the masked areas.
    '''
    
    #calculate the patch size
    num_patch_row = image_height//patch_size
    num_patch_col = image_width//patch_size
    
    
    #we flattened the image on the image height and width axis. We leave the channel axis.
    #NOTE that the flattened image is of [image_height*image_width, channel]
    #the masks array is calculated based on the patched images, not the original sized image.
    #flattened_image = image.reshape((-1, image_height*image_width)).detach() #since torch's image channel is in the first dimension.
    masks_array = masks_array.detach()
    
    masked_images = []
    #iterate through each mask
    for idx, mask in enumerate(masks_array):

         

        image_to_be_masked = image.clone() 
        if negate_mask: #initialize a full black image.
            image_to_be_masked = torch.zeros((image.size(0), image_height, image_width))

        for index in mask:
            row = index//num_patch_row
            col = (index % num_patch_col)
            
            if negate_mask:
                #fill in the original image's elements in the black image.
                image_to_be_masked[:, col*num_patch_col:col*num_patch_col+num_patch_col, row*num_patch_row:row*num_patch_row+num_patch_row] = image[:, col*num_patch_col:col*num_patch_col+num_patch_col, row*num_patch_row:row*num_patch_row+num_patch_row]
            else: 
                #block the masked area.
                image_to_be_masked[:, col*num_patch_col:col*num_patch_col+num_patch_col, row*num_patch_row:row*num_patch_row+num_patch_row] = 0.
        
        masked_image = torch.reshape(image_to_be_masked, (-1, image_height, image_width)).numpy()
        masked_image = np.transpose(masked_image, (1,2,0))
        #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        masked_images.append(masked_image)

    
    return masked_images
            








