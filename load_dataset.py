'''Dataset loading module.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import deeplake
from torchvision import transforms

from models.multiblock import MultiBlockMaskCollator
import cred

class DeepLakeDataset:
    '''Load desired dataset from deeplake API.
    '''


    def __init__(self, token, deeplake_dataset, batch_size, shuffle, mode='train'):
        '''Init parameters.
        '''

        self.token = token
        self.deeplake_dataset = deeplake_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode

    @staticmethod
    def image_transforms():

        return transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
#            transforms.RandomHorizontalFlip(),
#            transforms.ColorJitter(brightness=cfg.COLOR_JITTER_BRIGHTNESS, hue=cfg.COLOR_JITTER_HUE),
#            transforms.RandomAffine(degrees=cfg.RANDOM_AFFINE_ROTATION_RANGE, translate=cfg.RANDOM_AFFINE_TRANSLATE_RANGE, scale=cfg.RANDOM_AFFINE_SCALE_RANGE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1))
        ])


    def __call__(self):


        deeplake_data = deeplake.load(self.deeplake_dataset, token=self.token)

        dataloader = deeplake_data.dataloader().transform({'images':self.image_transforms, 'labels':None}).batch(self.batch_size).shuffle(self.shuffle).pytorch(collate_fn=MultiBlockMaskCollator(), decode_method={'images':'pil'})

        return dataloader


if __name__ == '__main__':
    m = DeepLakeDataset(token=cred.DEEPLAKE_TOKEN, deeplake_dataset='hub://activeloop/imagenet-train', batch_size=2, shuffle=False)

    print(m)
