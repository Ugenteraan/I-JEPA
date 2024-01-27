'''Training module for I-JEPA.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, dataset, DataLoader
from torchsummary import summary
import cv2
import argparse
import yaml
import copy


#import ijepa modules
from models.vit import VisionTransformerForEncoder as vitencoder 
from models.vit import VisionTransformerForPredictor as vitpredictor
from models.multiblock import MultiBlockMaskCollator
from load_dataset import LoadLocalDataset
from init_optim import InitOptimWithSGDR

def main(args):

    #Read the config file from args.
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        print("Configuration read successful...")

    #@@@@@@@@@@@@@@@@@@@@@@@@@ Extract the configurations from YAML file @@@@@@@@@@@@@@@@@@@@@@

    #Data configurations
    BATCH_SIZE = config['data']['batch_size']
    IMAGE_SIZE = config['data']['image_size']
    IMAGE_DEPTH = config['data']['image_depth']
    DATASET_FOLDER = config['data']['dataset_folder']
    NUM_WORKERS = config['data']['num_workers']
    SHUFFLE = config['data']['shuffle']
    USE_RANDOM_HORIZONTAL_FLIP = config['data']['use_random_horizontal_flip']
    RANDOM_AFFINE_DEGREES = config['data']['random_affine']['degrees']
    RANDOM_AFFINE_TRANSLATE = config['data']['random_affine']['translate']
    RANDOM_AFFINE_SCALE = config['data']['random_affine']['scale']
    COLOR_JITTER_BRIGHTNESS = config['data']['color_jitter']['brightness']
    COLOR_JITTER_HUE = config['data']['color_jitter']['hue']

    #Mask configurations
    ALLOW_OVERLAP = config['mask']['allow_overlap']
    PATCH_SIZE = config['mask']['patch_size']
    ASPECT_RATIO = config['mask']['aspect_ratio']
    NUM_CONTEXT_MASK = config['mask']['num_context_mask']
    NUM_PRED_TARGET_MASK = config['mask']['num_pred_target_mask']
    PRED_TARGET_MASK_SCALE = config['mask']['pred_target_mask_scale']
    CONTEXT_MASK_SCALE = config['mask']['context_mask_scale']
    MIN_MASK_LENGTH = config['mask']['min_mask_length']

    #Model configurations
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    TRANSFORMER_DEPTH = config['model']['transformer_depth']
    ENCODER_NETWORK_EMBEDDING_DIM = config['model']['encoder_network_embedding_dim']
    PREDICTOR_NETWORK_EMBEDDING_DIM = config['model']['predictor_network_embedding_dim']
    PROJECTION_KEYS_DIM = config['model']['projection_keys_dim']
    PROJECTION_VALUES_DIM = config['model']['projection_values_dim']
    FEEDFORWARD_PROJECTION_DIM = config['model']['feedforward_projection_dim']
    NUM_HEADS = config['model']['num_heads']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    FEEDFORWARD_DROPOUT_PROB = config['model']['feedforward_dropout_prob']

    #Training configurations
    DEVICE = config['training']['device']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    LOAD_CHECKPOINT = config['training']['load_checkpoint']
    END_EPOCH = config['training']['end_epoch']
    START_EPOCH = config['training']['start_epoch']
    COSINE_UPPER_BOUND_LR = config['training']['cosine_upper_bound_lr']
    COSINE_LOWER_BOUND_LR = config['training']['cosine_lower_bound_lr']
    WARMUP_START_LR = config['training']['warmup_start_lr']
    WARMUP_STEPS = config['training']['warmup_steps']
    NUM_EPOCH_TO_RESTART_LR = config['training']['num_epoch_to_restart_lr']
    COSINE_UPPER_BOUND_WD = config['training']['cosine_upper_bound_wd']
    COSINE_LOWER_BOUND_WD = config['training']['cosine_lower_bound_wd']
    USE_BFLOAT16 = config['training']['use_bfloat16']

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    
    #Init models
    print("Init encoder model....")
    ENCODER_NETWORK = vitencoder(image_size=IMAGE_SIZE,
                         patch_size=PATCH_SIZE, 
                         image_depth=IMAGE_DEPTH, 
                         encoder_network_embedding_dim=ENCODER_NETWORK_EMBEDDING_DIM, 
                         device=DEVICE,
                         transformer_network_depth=TRANSFORMER_DEPTH,
                         projection_keys_dim=PROJECTION_KEYS_DIM,
                         projection_values_dim=PROJECTION_VALUES_DIM,
                         num_heads=NUM_HEADS,
                         attn_dropout_prob=ATTN_DROPOUT_PROB,
                         feedforward_projection_dim=FEEDFORWARD_PROJECTION_DIM,
                         feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB)  

    summary(ENCODER_NETWORK, (IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE))
    
    print("Init predictor model...")
    PREDICTOR_NETWORK = vitpredictor(input_dim=ENCODER_NETWORK_EMBEDDING_DIM, #the input dim for the predictor network is the output dim from the encoder network.
                            predictor_network_embedding_dim=PREDICTOR_NETWORK_EMBEDDING_DIM, 
                            device=DEVICE,
                            transformer_network_depth=TRANSFORMER_DEPTH,
                            projection_keys_dim=PROJECTION_KEYS_DIM,
                            projection_values_dim=PROJECTION_VALUES_DIM,
                            num_heads=NUM_HEADS,
                            attn_dropout_prob=ATTN_DROPOUT_PROB,
                            feedforward_projection_dim=FEEDFORWARD_PROJECTION_DIM,
                            feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB)
    
    #to be used to generate the target embeddings. This network shares the same parameters as the encoder network.
    TARGET_ENCODER = copy.deepcopy(ENCODER_NETWORK) #creates an independent copy of the entire object hierarchy.    
    
    #initialize the mask collator module.
    MASK_COLLATOR_FN = MultiBlockMaskCollator(image_size=IMAGE_SIZE,
                                              patch_size=PATCH_SIZE,
                                              num_context_mask=NUM_CONTEXT_MASK,
                                              num_pred_target_mask=NUM_PRED_TARGET_MASK,
                                              context_mask_scale=CONTEXT_MASK_SCALE,
                                              pred_target_mask_scale=PRED_TARGET_MASK_SCALE,
                                              aspect_ratio=ASPECT_RATIO,
                                              min_mask_length=MIN_MASK_LENGTH,
                                              allow_overlap=ALLOW_OVERLAP)
    
    transforms_compose_list = [transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS, hue=COLOR_JITTER_HUE),
                          transforms.RandomAffine(degrees=RANDOM_AFFINE_DEGREES, translate=RANDOM_AFFINE_TRANSLATE, scale=RANDOM_AFFINE_SCALE),
                          transforms.ToTensor()
                          ]
    #insert the random horizontal flip to the list at the beginning if it's true.
    if USE_RANDOM_HORIZONTAL_FLIP:
        transforms_compose_list.insert(0, transforms.RandomHorizontalFlip())
    #insert the lambda function to convert grayscale images (with depth 1) to RGB (sort of) images. This is required since some images in dataset might be originally grayscale.
    if IMAGE_DEPTH == 3:
        transforms_compose_list.insert(-2, transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)))


    #dataloader init.
    DATASET_MODULE = LoadLocalDataset(dataset_folder_path=DATASET_FOLDER,
                                      transform=transforms.Compose(transforms_compose_list))

    DATASET_LOADER = DataLoader(DATASET_MODULE, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS, collate_fn=MASK_COLLATOR_FN) 
    
    #As for the optimizer, we can create one optimizer for each model or create one optimizer and pass in multiple models' params. 
    #we'll go with the 2nd option here.
    iterations_per_epoch = len(DATASET_LOADER)
    #this module contains the init for optimizer and schedulers.
    OPTIM_AND_SCHEDULERS = InitOptimWithSGDR(
                                             encoder_network=ENCODER_NETWORK, 
                                             predictor_network=PREDICTOR_NETWORK, 
                                             cosine_upper_bound_lr=COSINE_UPPER_BOUND_LR, 
                                             cosine_lower_bound_lr=COSINE_LOWER_BOUND_LR, 
                                             warmup_start_lr=WARMUP_START_LR, 
                                             warmup_steps=WARMUP_STEPS,
                                             num_steps_to_restart_lr=NUM_EPOCH_TO_RESTART_LR*iterations_per_epoch,
                                             cosine_upper_bound_wd=COSINE_UPPER_BOUND_WD,
                                             cosine_lower_bound_wd=COSINE_LOWER_BOUND_WD
                                            ) 
    OPTIMIZER = OPTIM_AND_SCHEDULERS.get_optimizer()
    

    for epoch_idx in range(START_EPOCH, END_EPOCH):
        print("here")
























if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    args = parser.parse_args()

    main(args)

