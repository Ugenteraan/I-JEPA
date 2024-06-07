

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
from PIL import Image, ImageOps
import numpy as np
import cv2
import torch
from torch.utils import data
from torch.utils.data import Dataset, dataset, DataLoader
from torchvision import transforms




class LoadUnlabelledDataset(Dataset):
    '''Loads the dataset from the given path.
    '''

    def __init__(self, dataset_folder_path, image_size=224, image_depth=3, transform=None, logger=None):
        '''Parameter Init.
        '''

        if dataset_folder_path is None:
            logger.error("Dataset folder path must be provided!")
            sys.exit()

        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.image_size = image_size
        self.image_depth = image_depth
        self.image_path = self.read_folder()
        self.logger = logger



    def read_folder(self):
        '''Reads the folder for the images.
        '''
        
        image_path = []
    
        folder_path = f"{self.dataset_folder_path.rstrip('/')}/"

        for x in glob.glob(folder_path + "**", recursive=True):

            if not x.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            image_path.append(x)

        return image_path


    def __len__(self):
        '''Returns the total size of the data.
        '''
        return len(self.image_path)

    def __getitem__(self, idx):
        '''Returns a single image and its corresponding label.
        '''

      if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_path[idx]


        try:

            image = Image.open(image_path).convert('RGB')

        except Exception as err:
            if self.logger is not None:
                self.logger.error(f"{image_path}")
                self.logger.error(f"Error loading image: {err}")
            sys.exit()

        image = image.resize((self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return {
            'images': image
        }


class LoadLabelledDataset(Dataset):
    '''Loads the dataset from the given path. 
    '''

    def __init__(self, dataset_folder_path, image_size=224, image_depth=3, train=True, transform=None, logger=None):
        '''Parameter Init.
        '''

        assert not dataset_folder_path is None, "Path to the dataset folder must be provided!"

        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.image_size = image_size
        self.image_depth = image_depth
        self.train = train
        self.classes = sorted(self.get_classnames())
        self.image_path_label = self.read_folder()
        self.logger = logger


    def get_classnames(self):
        '''Returns the name of the classes in the dataset.
        '''
        return os.listdir(f"{self.dataset_folder_path.rstrip('/')}/train/" )


    def read_folder(self):
        '''Reads the folder for the images with their corresponding label (foldername).
        '''

        image_path_label = []

        if self.train:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/train/"
        else:
            folder_path = f"{self.dataset_folder_path.rstrip('/')}/test/"

        for x in glob.glob(folder_path + "**", recursive=True):

            if not x.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            class_idx = self.classes.index(x.split('/')[-2])
            image_path_label.append((x, int(class_idx)))

        return image_path_label


    def __len__(self):
        '''Returns the total size of the data.
        '''
        return len(self.image_path_label)

    def __getitem__(self, idx):
        '''Returns a single image and its corresponding label.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.image_path_label[idx]

        try:
            if self.image_depth == 1:
                image = cv2.imread(image_path, 0)
            else:
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            

            #sometimes PIl throws truncated image error. Perhaps due to the image being too big? Hence the cv2 imread.
            image = Image.fromarray(image)

        except Exception as err:
            self.logger.error(f"Error loading image: {err}")
            sys.exit()

        image = image.resize((self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }



'''
if __name__ == '__main__':

        import time
        from models.vit import VisionTransformerForPredictor as vit 
        from models.vit import VisionTransformerForEncoder as vitencoder 
        from models.multiblock import MultiBlockMaskCollator



        load_dataset_module = LoadLocalDataset(dataset_folder_path="/home/topiarypc/Projects/Attention-CNN-Visualization/image_dataset/", transform=transforms.ToTensor())

        dataloader = DataLoader(load_dataset_module, batch_size=3, shuffle=False, num_workers=4, collate_fn=MultiBlockMaskCollator())
        
        device = torch.device('cuda:0')
        v = vit(input_dim=512, predictor_network_embedding_dim=512, projection_keys_dim=512, projection_values_dim=512, feedforward_projection_dim=512, transformer_network_depth=5, num_heads=8, attn_dropout_prob=0.1, feedforward_dropout_prob=0.1, device=device)
        v_encoder = vitencoder(image_size=224, image_depth=3, patch_size=14, in_channel=3, encoder_network_embedding_dim=512, feedforward_projection_dim=512, transformer_network_depth=5, num_heads=8, device=device, attn_dropout_prob=0.1, feedforward_dropout_prob=0.1, projection_keys_dim=512, projection_values_dim=512)

        
        start_ = time.time() 
        for idx, data in enumerate(dataloader):

            images = data['collated_batch_data_images'].to(device)
            masks_pred_target = torch.tensor(np.asarray(data['collated_masks_pred_target']), dtype=torch.int64, device=device)
            masks_ctxt = torch.tensor(np.asarray(data['collated_masks_context']), dtype=torch.int64, device=device)


            y = v_encoder(images, masks_ctxt)

            print("Y: ", y.size(), masks_pred_target.size(), masks_ctxt.size())
            x = v(y,  masks_ctxt=masks_ctxt, masks_pred_target=masks_pred_target)
            print("The big size: ", x.size())

            

            if idx == 10: 
                pass
        end_ = time.time()

        print("Time taken: ", end_ - start_)
'''
