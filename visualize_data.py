'''Module to visualize the dataset.
'''


from load_dataset import DeepLakeDataset
import matplotlib.pyplot as plt

import cred
from utils import apply_masks_over_image_patches


class VisualizeData:

    def __init__(self, deeplake_module, deeplake_dataset, visualize_batch_size, visualize_shuffle, deeplake_token, num_images):

        self.dataloader = deeplake_module(token=deeplake_token, deeplake_dataset=deeplake_dataset, batch_size=visualize_batch_size, shuffle=visualize_shuffle)()
    
    def apply_masks_over_images(self, image, masks):
        '''
        '''

    def __call__(self):
        

        for idx, batch_data in enumerate(self.dataloader):
            
            images = batch_data[0]
            labels = batch_data[1]
            pred_target_masks = batch_data[2]
            context_masks = batch_data[3]
            
            apply_masks_over_image_patches(image=images[0], patch_size=14, image_height=224, image_width=224, masks_array=pred_target_masks[0]) 

            
            


m = DeepLakeDataset(token=cred.DEEPLAKE_TOKEN, deeplake_dataset='hub://activeloop/imagenet-train', batch_size=2, shuffle=True)()

if __name__ == '__main__':
    vd = VisualizeData(deeplake_module = DeepLakeDataset, deeplake_dataset='hub://activeloop/imagenet-train', visualize_batch_size=3, visualize_shuffle=False, deeplake_token=cred.DEEPLAKE_TOKEN, num_images=10)

    print(vd())
