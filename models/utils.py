'''Helper functions.
'''


import torch

def apply_masks(x, masks):
    '''
    x: tensor [batch size, num patches, embedding dim]
    masks: LIST of tensors containing indices of patches (2nd dimension of x) to keep

    returns the image patches at the indices of the masks only. 
    e.g. [batch size, 3, embedding dim]. 3 is the total number of indices in one of the mask in the masks list.
    '''


    all_masked_patch_embeddings = []

    for m in masks:

        print(m.size())


    
