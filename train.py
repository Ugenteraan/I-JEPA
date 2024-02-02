'''Training module for I-JEPA.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, dataset, DataLoader
from torchsummary import summary
import numpy as np
import argparse
import yaml
import copy
from loguru import logger
import datetime


#import ijepa modules
from models.vit import VisionTransformerForEncoder as vitencoder 
from models.vit import VisionTransformerForPredictor as vitpredictor
from models.multiblock import MultiBlockMaskCollator
from load_dataset import LoadLocalDataset
from init_optim import InitOptimWithSGDR
from utils import apply_masks_over_embedded_patches, repeat_interleave_batch, loss_fn, load_checkpoint, save_checkpoint

DATETIME_NOW = datetime.datetime.now().replace(second=0, microsecond=0) #datetime without seconds & miliseconds.

def train(args):

    #Read the config file from args.
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        print("Configuration read successful...")
    
    with open(args.logging_config, 'r') as logging_configfile:
        logging_config = yaml.load(logging_configfile, Loader=yaml.FullLoader)
        print("Logging configuration file read successful...")
        print("Initializing logger")

    ###Logger initialization
    if logging_config['disable_default_loggers']:
        logger.remove(0)

    logging_formatter = logging_config['formatters'][config['env']] #set the environment for the logger. 
    
    #to output to a file
    logger.add(f"{logging_config['log_dir']}{DATETIME_NOW}-{logging_config['log_filename']}",
                    level=logging_formatter['level'], 
                    format=logging_formatter['format'],
                    backtrace=logging_formatter['backtrace'],
                    diagnose=logging_formatter['diagnose'],
                    enqueue=logging_formatter['enqueue'])

    #to output to the console.
    logger.add(sys.stdout,
                level=logging_formatter['level'], 
                format=logging_formatter['format'],
                backtrace=logging_formatter['backtrace'],
                colorize=True,
                diagnose=logging_formatter['diagnose'],
                enqueue=logging_formatter['enqueue'])

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
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
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
    LOAD_CHECKPOINT_EPOCH = config['training']['load_checkpoint_epoch']
    EMA = config['training']['ema']
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
    
    #calculating the number of patches for the initialization of parameter in the predictor network.
    NUM_PATCHES = (IMAGE_SIZE/PATCH_SIZE)**2
    
    #Init models
    logger.info("Init Encoder model...")
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
    
    logger.info("Init Predictor model...")
    PREDICTOR_NETWORK = vitpredictor(input_dim=ENCODER_NETWORK_EMBEDDING_DIM, #the input dim for the predictor network is the output dim from the encoder network.
                            predictor_network_embedding_dim=PREDICTOR_NETWORK_EMBEDDING_DIM, 
                            num_patches=NUM_PATCHES,
                            device=DEVICE,
                            transformer_network_depth=TRANSFORMER_DEPTH,
                            projection_keys_dim=PROJECTION_KEYS_DIM,
                            projection_values_dim=PROJECTION_VALUES_DIM,
                            num_heads=NUM_HEADS,
                            attn_dropout_prob=ATTN_DROPOUT_PROB,
                            feedforward_projection_dim=FEEDFORWARD_PROJECTION_DIM,
                            feedforward_dropout_prob=FEEDFORWARD_DROPOUT_PROB)
    
    #to be used to generate the target embeddings. This network shares the same parameters as the encoder network.
    TARGET_ENCODER_NETWORK = copy.deepcopy(ENCODER_NETWORK) #creates an independent copy of the entire object hierarchy.    

    
    #initialize the mask collator module.
    logger.info("Init MultiBlockMaskCollator module...")
    MASK_COLLATOR_FN = MultiBlockMaskCollator(image_size=IMAGE_SIZE,
                                              patch_size=PATCH_SIZE,
                                              num_context_mask=NUM_CONTEXT_MASK,
                                              num_pred_target_mask=NUM_PRED_TARGET_MASK,
                                              context_mask_scale=CONTEXT_MASK_SCALE,
                                              pred_target_mask_scale=PRED_TARGET_MASK_SCALE,
                                              aspect_ratio=ASPECT_RATIO,
                                              min_mask_length=MIN_MASK_LENGTH,
                                              allow_overlap=ALLOW_OVERLAP,
                                              logger=logger)
    
    transforms_compose_list = [transforms.ColorJitter(brightness=COLOR_JITTER_BRIGHTNESS, hue=COLOR_JITTER_HUE),
                          transforms.RandomAffine(degrees=RANDOM_AFFINE_DEGREES, translate=RANDOM_AFFINE_TRANSLATE, scale=RANDOM_AFFINE_SCALE),
                          transforms.ToTensor()
                          ]
    #insert the random horizontal flip to the list at the beginning if it's true.
    if USE_RANDOM_HORIZONTAL_FLIP:
        transforms_compose_list.insert(0, transforms.RandomHorizontalFlip())
    #insert the lambda function to convert grayscale images (with depth 1) to RGB (sort of) images. This is required since some images in dataset might be originally grayscale.
    if IMAGE_DEPTH == 3:
        #this process should be AFTER the image has been converted to tensor.
        transforms_compose_list.append(transforms.Lambda(lambda x: x.repeat(int(3/x.shape[0]), 1, 1)))


    #dataloader init.
    logger.info("Init local dataset loading module...")
    DATASET_MODULE = LoadLocalDataset(dataset_folder_path=DATASET_FOLDER,
                                      transform=transforms.Compose(transforms_compose_list),
                                      logger=logger)

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
                                             cosine_lower_bound_wd=COSINE_LOWER_BOUND_WD,
                                             logger=logger
                                            ) 
    OPTIMIZER = OPTIM_AND_SCHEDULERS.get_optimizer()
    SCALER = None
    
    #generator object that holds momentum scaling values (to control the strength of the momentum) from the start of the training to the end. 
    MOMENTUM_SCHEDULER = (EMA[0] + i*(EMA[1] - EMA[0])/(iterations_per_epoch*(END_EPOCH-START_EPOCH)) for i in range(int(END_EPOCH - START_EPOCH)*iterations_per_epoch+1))

    #scaler is used to scale the values in variables like state_dict, optimizer etc to bfloat16 type.
    if USE_BFLOAT16:
        SCALER = torch.cuda.amp.GradScaler()
    
    
    if LOAD_CHECKPOINT:
        ENCODER_NETWORK, PREDICTOR_NETWORK, TARGET_ENCODER_NETWORK, OPTIMIZER, SCALER, START_EPOCH = load_checkpoint(encoder_network=ENCODER_NETWORK,
                                                                                            predictor_network=PREDICTOR_NETWORK,
                                                                                            target_encoder_network=TARGET_ENCODER_NETWORK,
                                                                                            model_save_folder=MODEL_SAVE_FOLDER,
                                                                                            model_name=MODEL_NAME,
                                                                                            optimizer=OPTIMIZER,
                                                                                            scaler=SCALER,
                                                                                            load_checkpoint_epoch=LOAD_CHECKPOINT_EPOCH, 
                                                                                            logger=logger)
     
    #set the models to train mode.
    ENCODER_NETWORK.train()
    PREDICTOR_NETWORK.train()
  

    for epoch_idx in range(START_EPOCH, END_EPOCH):
        logger.info(f"Training has started for epoch {epoch_idx}")
        
        epoch_loss = 0

        for idx, data in enumerate(DATASET_LOADER):

            images = data['collated_batch_data_images'].to(DEVICE)
            masks_pred_target = torch.tensor(np.asarray(data['collated_masks_pred_target']), dtype=torch.int64, device=DEVICE)
            masks_ctxt = torch.tensor(np.asarray(data['collated_masks_context']), dtype=torch.int64, device=DEVICE)

            batch_size = len(images)
    
            #forward propagation.
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):
                
                #first we want to create the target. With no gradients involved.
                with torch.no_grad():   
                    actual_target_embeddings  = TARGET_ENCODER_NETWORK(images)
                    actual_target_embeddings = apply_masks_over_embedded_patches(actual_target_embeddings, masks_pred_target)
                    actual_target_embeddings = repeat_interleave_batch(actual_target_embeddings, batch_size, NUM_CONTEXT_MASK) #in case the number of context mask is more than 1.

                #create the embedding for the context image using the encoder network.
                context_image_embedding = ENCODER_NETWORK(images, masks_ctxt)

                #perform prediction.
                predicted_target_embeddings = PREDICTOR_NETWORK(x=context_image_embedding, masks_ctxt=masks_ctxt, masks_pred_target=masks_pred_target)    

                #calculate loss
                loss = loss_fn(prediction=predicted_target_embeddings, target=actual_target_embeddings)
                epoch_loss += loss.item()

            #backward and step
            if USE_BFLOAT16:
                SCALER.scale(loss).backward()
                SCALER.step(OPTIMIZER)
                SCALER.update()
            else:
                loss.backward()
                OPTIMIZER.step()

            OPTIMIZER.zero_grad()
            _new_lr, _new_wd = OPTIM_AND_SCHEDULERS.step()
            #once the gradients are updated, the target encoder has to be updated with the new weight. 
            #the update will be done with a momentum parameter. That is, the old network parameter will be scaled with a momentum parameter and then multiplied to the new network parameter.
            #we will be using momentum scheduler to control the momentum weight.
            with torch.no_grad():
                m = next(MOMENTUM_SCHEDULER)
                for param_q, param_k in zip(ENCODER_NETWORK.parameters(), TARGET_ENCODER_NETWORK.parameters()):
                    
                    #we want to multiply the parameters from target encoder with the momentum scaling and then add a small value (1 - momentum value) from the encoder network.
                    param_k.mul_(m).add((1.-m) * param_q.detach().data) #update the target encoder's parameter.

        logger.info(f"The loss at epoch {epoch_idx} is {epoch_loss}") 

        #save checkpoints as per defined in the model save frequency (epoch).
        if epoch_idx % MODEL_SAVE_FREQ == 0:
            
            #save the predictor network
            save_checkpoint(model_save_folder=MODEL_SAVE_FOLDER, 
                            model_name=MODEL_NAME, 
                            encoder_network=ENCODER_NETWORK, 
                            predictor_network=PREDICTOR_NETWORK, 
                            target_encoder_network=TARGET_ENCODER_NETWORK, 
                            optimizer=OPTIMIZER, 
                            scaler=SCALER, 
                            epoch=epoch_idx, 
                            loss=loss,
                            logger=logger)
            



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    parser.add_argument('--logging_config', required=True, type=str, help='Specify the YAML config file to be used for the logging module.')
    args = parser.parse_args()

    train(args)

