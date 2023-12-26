'''Helper functions to be used throughout the project.
'''

import torch
import numpy as np



def apply_masks_over_image_patches(image, patch_size, image_height, image_width, masks_array):
    '''Applies patched masks on the original image and returns the resulting image.
    '''
    
    #calculate the patch size
    patch_height = image_height//patch_size
    patch_width = image_width//patch_size
    
    
    #we flattened the image on the image height and width axis. We leave the channel axis.
    #NOTE that the flattened image is of [image_height*image_width, channel]
    #the masks array is calculated based on the patched images, not the original sized image.
    
    flattened_image = image.reshape((image_height*image_width, -1))
    
    masked_images = []
    #iterate through each mask
    for mask in masks_array:
        
        for index in mask:
            
            mask[index*patch_size-patch_size:index*patch_size, 3] = 0

        masked_images.append(mask)

    
    return masked_images
            








