'''Helper functions to be used throughout the project.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def apply_masks_over_embedded_patches(x, masks):
    '''
    x: tensor [batch size, num patches, embedding dim]
    masks: LIST of tensors containing indices of patches (2nd dimension of x) to keep

    returns the image patches at the indices of the masks only. 
    e.g. [batch size, 3, embedding dim]. 3 is the total number of indices in one of the mask in the masks list.
    '''


    all_masked_patch_embeddings = []
    
    for idx, m in enumerate(masks):
        #m is of size [batch size, index of patch]
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1)) #Reshape the mask tensor to be compatible with the given input tensor.
        
        #collect all the tensors in dimension 1 (number of patch dim) based on the given mask and append to the list.
        all_masked_patch_embeddings += [torch.gather(x, dim=1, index=mask_keep)]
    
        
    return torch.cat(all_masked_patch_embeddings, dim=0) 




def apply_masks_over_image_patches(image, patch_size, image_size,  masks_array, batch_idx, negate_mask=True):
    '''Applies patched masks on the original image and returns the resulting image.
       negate_mask parameter is used to reverse the mask's purpose. That is to only show the masked areas instead of block the masked areas.
    '''
    
    #calculate the number of patches in both sides.
    num_patches = image_size//patch_size
    
    masked_images = []
    
    #since the masks are in [mask num, batch size, mask indices] shape, we gotta transpose the first two dimensions so we can iterate through ALL masks in a single batch.
    mask_array = torch.from_numpy(np.asarray(masks_array)).transpose(0,1)
    mask_array = mask_array[batch_idx] #get all the masks belonging to the specified batch.

    for idx, mask in enumerate(mask_array):
        '''The idea here is to iterate through the masks in a batch, get all the indices and convert the PATCH indices to pixel-level indices and apply the maskings.
        '''

        image_to_be_masked = image.clone() 
        if negate_mask: #initialize a full black image.
            image_to_be_masked = torch.zeros((image.size(0), image_size, image_size))

        for index in mask:
            '''For reference: 
                patch_size = 14
                num_patches = 16
            '''
            
            #first we have to find the row and column of the patch.
            row = index//num_patches
            col = (index % num_patches)

            
            if negate_mask:
                #fill in the original image's elements in the black image.
                image_to_be_masked[:, col*num_patches:col*num_patches+num_patches, row*num_patches:row*num_patches+num_patches] = image[:, col*num_patches:col*num_patches+num_patches, row*num_patches:row*num_patches+num_patches]
            else: 
                #block the masked area.
                image_to_be_masked[:, col*num_patches:col*num_patches+num_patches, row*num_patches:row*num_patches+num_patches] = 0.


        masked_image = torch.reshape(image_to_be_masked, (-1, image_size, image_size)).numpy()
        masked_image = np.transpose(masked_image, (1,2,0))
        #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        masked_images.append(masked_image)

    
    return masked_images


def repeat_interleave_batch(x, batch_size, repeat):
    '''This function is the same as torch's repeat interleave, except that this works only on the batch dimension.
       Basically here, we're "stacking" each element 'repeat' times in its place rather than "stacking" the entire tensor on top of each other 'repeat' times.
    '''
    N = len(x) // batch_size
    x = torch.cat([
        torch.cat([x[i*batch_size:(i+1)*batch_size] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x



def loss_fn(prediction, target):
    '''The loss function of I-Jepa is basically just L1 loss.
    '''

    loss = F.smooth_l1_loss(prediction, target)

    return loss


def save_checkpoint(model_save_folder, model_name, encoder_network, predictor_network, target_encoder_network, optimizer, scaler, epoch, loss, logger=None):
    '''Save model checkpoint.
    '''
    save_dict = {
                'encoder_network': encoder_network.state_dict(),
                'predictor_network': predictor_network.state_dict(),
                'target_encoder_network': target_encoder_network.state_dict(),
                'optimizer': optimizer,
                'scaler': scaler, 
                'epoch': epoch, #useful for resuming training from the last epoch.
                'loss' : loss #record purposes. 
                }
    
    try:
        torch.save(save_dict, f"{model_save_folder.rstrip('/')}/{model_name}-checkpoint-ep-{epoch}.pth.tar") 
        torch.save(save_dict, f"{model_save_folder.rstrip('/')}/{model_name}-latest.pth.tar") 
        logger.info(f"Model checkpoint save for epoch {epoch} is successful!")
    except Exception as err:
        logger.error(f"Model checkpoint save for epoch {epoch} has failed! {err}")

    return None




def load_checkpoint(model_save_folder, model_name, encoder_network, predictor_network, target_encoder_network, optimizer, scaler, load_checkpoint_epoch=None, logger=None):
    '''Loads either the latest model (if load_checkpoint_val is None) or loads the specific checkpoint.
    '''

    try:
        checkpoint = None 
        if not load_checkpoint_epoch is None:
            checkpoint = torch.load(f"{model_save_folder.rstrip('/')}/{model_name}-checkpoint-ep-{load_checkpoint_epoch}.pth.tar")
        else:
            checkpoint = torch.load(f"{model_save_folder.rstrip('/')}/{model_name}-latest.pth.tar")

        epoch = checkpoint['epoch']
        logger.info("Checkpoint from epoch {epoch} is successfully loaded! Extracting the parameters to load to individual model/variabels now...")

        msg = encoder_network.load_state_dict(checkpoint['encoder_network'])
        logger.info(f"Loaded pretrained encoder network with msg: {msg}")

        msg = predictor_network.load_state_dict(checkpoint['predictor_network'])
        logger.info(f"Loaded pretrained predictor network with msg: {msg}")

        if target_encoder_network is not None:
            msg = target_encoder_network.load_state_dict(checkpoint['target_encoder_network'])
            logger.info(f"Loaded target encoder network with msg: {msg}")

        optimizer.load_state_dict(checkpoint['optimizer'])

        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])    

        logger.info(f"Loaded optimizers and scalers from checkpoint...")

    
    except Exception as err:
        logger.error(f"Error loading the model! {err}")
        epoch = 0

    return encoder_network, predictor_network, target_encoder_network, optimizer, scaler, epoch
        


            

def load_checkpoint_downstream(trained_model_folder, trained_model_name, encoder_network, predictor_network, load_checkpoint_epoch=None, strict=False, logger=None):
    '''Loads either the latest model (if load_checkpoint_val is None) or loads the specific checkpoint.
    '''

    try:
        checkpoint = None 
        if not load_checkpoint_epoch is None:
            checkpoint = torch.load(f"{trained_model_folder.rstrip('/')}/{trained_model_name}-checkpoint-ep-{load_checkpoint_epoch}.pth.tar")
        else:
            checkpoint = torch.load(f"{trained_model_folder.rstrip('/')}/{trained_model_name}-latest.pth.tar")


        msg = encoder_network.load_state_dict(checkpoint['encoder_network'], strict=strict)
        logger.info(f"Loaded pretrained encoder network with msg: {msg}")

        msg = predictor_network.load_state_dict(checkpoint['predictor_network'], strict=strict)
        logger.info(f"Loaded pretrained predictor network with msg: {msg}")


    
    except Exception as err:
        logger.error(f"Error loading the model! {err}")
        

    return encoder_network, predictor_network



def calculate_accuracy(predicted, target):
    '''Calculates the accuracy of the prediction.
    '''


    num_data = target.size()[0]
    predicted = torch.argmax(predicted, dim=1)

    correct_pred = torch.sum(predicted == target)

    accuracy = correct_pred*(num_data/100)

    return accuracy.item()




