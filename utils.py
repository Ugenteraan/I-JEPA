'''Helper functions to be used throughout the project.
'''

import cv2
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
    
    print(image.size())
    flattened_image = image.reshape((-1, image_height*image_width)) #since torch's image channel is in the first dimension.
    
    masked_images = []
    #iterate through each mask
    for mask in masks_array:
        
        print("mask: ", mask)
        for index in mask:
            
            flattened_image[:, index*patch_size-patch_size:index*patch_size] = 0
        
        masked_image = torch.reshape(flattened_image, (-1, image_height, image_width)).numpy()
        masked_image = np.transpose(masked_image, (1,2,0))
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("masked_img", masked_image)
        cv2.waitKey(0)
        masked_images.append(masked_image)

    
    return masked_images
            








