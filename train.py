'''Training module for I-JEPA.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torchsummary import summary
import cv2


def main():
    '''Main training module.
    '''

    img = cv2.imread('test.jpg')

    img_tensor = torch.from_numpy(cv2.resize(img, (224,224)))
    #img = torch.permute(img, (2, 0, 1)).to(torch.float32)
    print(img_tensor.size(), img_tensor.dtype)

    #unfolding_func = nn.Unfold(kernel_size=(16,16), stride=(16,16))
    
    #patched_image_tensors = unfolding_func(img)

    #print(patched_image_tensors.size())
    
    mask = torch.zeros((224,224, 3), dtype=torch.uint8) 
    

    #masked_img = torch.bitwise_and(img.type(torch.int8), mask)
    masked_img = img_tensor * mask

    numpy_img = masked_img.numpy()
    cv2.imshow('img', numpy_img)
    cv2.waitKey(0)



if __name__=='__main__':
    main()

