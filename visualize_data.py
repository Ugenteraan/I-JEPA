'''Module to visualize the dataset.
'''

import torch
import cv2
import numpy as np
from load_dataset import DeepLakeDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import cred
from utils import apply_masks_over_image_patches
from models.multiblock import MultiBlockMaskCollator
from dataload_manual import LoadDataset
from torchvision import transforms

class VisualizeData:

    def __init__(self, deeplake_module, deeplake_dataset, visualize_batch_size, visualize_shuffle, deeplake_token, image_height, image_width, patch_size, num_figs, fig_savepath):

        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.num_figs = num_figs
        self.fig_savepath = fig_savepath
        self.visualize_batch_size = visualize_batch_size
        #self.dataloader = deeplake_module(token=deeplake_token, collate_func=MultiBlockMaskCollator(), deeplake_dataset=deeplake_dataset, batch_size=visualize_batch_size, shuffle=visualize_shuffle)()
        self.load_dataset_module = LoadDataset(dataset_folder_path="/home/topiarypc/Projects/Attention-CNN-Visualization/image_dataset/", transform=transforms.ToTensor())
        self.dataloader = DataLoader(self.load_dataset_module, batch_size=visualize_batch_size, shuffle=visualize_shuffle, num_workers=8, collate_fn=MultiBlockMaskCollator())

   
    def plot_images(self, fig, axes, row_idx, original_image, masked_pred_images, masked_context_image):
        '''Plot images using matplotlib.
        '''
        
        #plot the original image 
        plt.sca(axes[row_idx, 0])
        plt.imshow(original_image)
        plt.axis('off')
        
        for i in range(len(masked_pred_images)):
            plt.sca(axes[row_idx, i+1])
            plt.imshow(masked_pred_images[i])
            plt.axis('off')

        plt.sca(axes[row_idx, 5])
        plt.imshow(masked_context_image[0])
        plt.axis('off')



    def __call__(self):
        
        
        fig, axes = plt.subplots(nrows=self.visualize_batch_size, ncols=6)

        for idx, batch_data in enumerate(self.dataloader):
            

            images = batch_data[0]
            labels = batch_data[1]
            pred_target_masks = batch_data[2]
            context_masks = batch_data[3]
            
            for batch_idx in range(self.visualize_batch_size):

                masked_pred_images = apply_masks_over_image_patches(image=images[batch_idx], patch_size=self.patch_size, image_height=self.image_height, image_width=self.image_width, masks_array=pred_target_masks[batch_idx]) 
                masked_context_image = apply_masks_over_image_patches(image=images[batch_idx], patch_size=self.patch_size, image_height=self.image_height, image_width=self.image_width, masks_array=context_masks[batch_idx])
                
                orig_image = torch.reshape(images[batch_idx], (-1, self.image_height, self.image_width)).numpy()
                orig_image = np.transpose(orig_image, (1,2,0))

                self.plot_images(fig=fig, axes=axes, row_idx=batch_idx, original_image=orig_image, masked_pred_images=masked_pred_images, masked_context_image=masked_context_image)
                

            plt.savefig(f'{self.fig_savepath}/visualization - {idx}.jpg')

            if idx == self.num_figs:
                break        



if __name__ == '__main__':
    import time

    start_ = time.time()
    vd = VisualizeData(deeplake_module = DeepLakeDataset, deeplake_dataset='hub://activeloop/imagenet-train', visualize_batch_size=5, visualize_shuffle=True, deeplake_token=cred.DEEPLAKE_TOKEN, num_figs=20, image_height=224, image_width=224, patch_size=14, fig_savepath='./figures/')

    print(vd())
    end_ = time.time()
    print("Time takes : ", end_ - start_)
