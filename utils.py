'''Helper functions to be used throughout the project.
'''

import cv2
import torch
import numpy as np


def apply_masks_over_embedded_patches(x, masks):
    '''
    x: tensor [batch size, num patches, embedding dim]
    masks: LIST of tensors containing indices of patches (2nd dimension of x) to keep

    returns the image patches at the indices of the masks only. 
    e.g. [batch size, 3, embedding dim]. 3 is the total number of indices in one of the mask in the masks list.
    '''


    all_masked_patch_embeddings = []
    
    #official i-jepa code only takes in 1 x and 1 mask batch at a time.
    for idx, m in enumerate(masks):
        #mask is of size [batch size, index of patch]
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1)) #Reshape the mask tensor to be compatible with the given input tensor.
        print(mask_keep.size())

        print(x.size())
        
        #collect all the tensors in dimension 1 (number of patch dim) based on the given mask and append to the list.
        all_masked_patch_embeddings += [torch.gather(x, dim=1, index=mask_keep)]
    
    return torch.cat(all_masked_patch_embeddings, dim=0)



    

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
            








