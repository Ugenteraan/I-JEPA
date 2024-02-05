'''Module to visualize the dataset.
'''

import torch
import cv2
import numpy as np
from deeplake_dataset import DeepLakeDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import cred
from utils import apply_masks_over_image_patches
from models.multiblock import MultiBlockMaskCollator
from load_dataset import LoadUnlabelledDataset
from torchvision import transforms

class VisualizeData:

    def __init__(self, visualize_batch_size=6, pred_masks_num=4, context_masks_num=1, num_workers=4, visualize_shuffle=False, image_size=224,  patch_size=16, num_figs=10, fig_savepath='./figures/', deeplake_module=None, deeplake_dataset=None, deeplake_token=None, dataset_folder_path=None, collate_func=None):

        self.image_size = image_size 
        self.patch_size = patch_size
        self.num_figs = num_figs
        self.fig_savepath = fig_savepath
        self.visualize_batch_size = visualize_batch_size
        self.pred_masks_num = pred_masks_num
        self.context_masks_num = context_masks_num

    
        self.dataloader = None
        if deeplake_module is not None and deeplake_dataset is not None and deeplake_token is not None:
            #multiple number of workers seems to be throwing errors when using Deeplake's dataloader.
            self.dataloader = deeplake_module(token=deeplake_token, collate_func=collate_func, deeplake_dataset=deeplake_dataset, batch_size=visualize_batch_size, shuffle=visualize_shuffle)()
        else:
            self.load_dataset_module = LoadUnlabelledDataset(dataset_folder_path=dataset_folder_path, image_size=image_size, transform=transforms.ToTensor())
            self.dataloader = DataLoader(self.load_dataset_module, batch_size=visualize_batch_size, shuffle=visualize_shuffle, num_workers=num_workers, collate_fn=collate_func)

   
    def plot_images(self, fig, axes, row_idx, original_image, masked_pred_images, masked_context_image):
        '''Plot images using matplotlib.
        '''
        
        #plot the original image 
        plt.sca(axes[row_idx, 0])
        plt.imshow(original_image)
        plt.axis('off')
        
        for i in range(self.pred_masks_num):
            plt.sca(axes[row_idx, i+1])
            plt.imshow(masked_pred_images[i])
            plt.axis('off')

        plt.sca(axes[row_idx, 5])
        plt.imshow(masked_context_image[0])
        plt.axis('off')



    def __call__(self):
        
        
        fig, axes = plt.subplots(nrows=self.visualize_batch_size, ncols=6)

        for idx, batch_data in enumerate(self.dataloader):
            

            images = batch_data['collated_batch_data_images']
            pred_target_masks = batch_data['collated_masks_pred_target']
            context_masks = batch_data['collated_masks_context']
            
            for batch_idx in range(self.visualize_batch_size):

                masked_pred_images = apply_masks_over_image_patches(image=images[batch_idx], patch_size=self.patch_size, image_size=self.image_size, masks_array=pred_target_masks, batch_idx=batch_idx) 
                masked_context_image = apply_masks_over_image_patches(image=images[batch_idx], patch_size=self.patch_size, image_size=self.image_size, masks_array=context_masks, batch_idx=batch_idx)
                
                orig_image = torch.reshape(images[batch_idx], (-1, self.image_size, self.image_size)).numpy()
                orig_image = np.transpose(orig_image, (1,2,0))

                self.plot_images(fig=fig, axes=axes, row_idx=batch_idx, original_image=orig_image, masked_pred_images=masked_pred_images, masked_context_image=masked_context_image)
                

            plt.savefig(f'{self.fig_savepath}/visualization - {idx}.jpg')

            if idx == self.num_figs:
                break        



if __name__ == '__main__':
    #vd = VisualizeData(deeplake_module = DeepLakeDataset, deeplake_dataset='hub://activeloop/imagenet-train', visualize_batch_size=6, visualize_shuffle=False, deeplake_token=cred.DEEPLAKE_TOKEN, num_figs=50, image_height=224, image_width=224, patch_size=14, fig_savepath='./figures/')
    print("Executing...")
    vd = VisualizeData(visualize_batch_size=4, visualize_shuffle=False, pred_masks_num=4, context_masks_num=1, num_workers=4, num_figs=10, image_size=224, patch_size=14, fig_savepath='./figures/', dataset_folder_path="./dog_breed_classification/ssl_train/", collate_func=MultiBlockMaskCollator())

    v = vd()
