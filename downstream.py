'''Module to downstream a trained I-JEPA's predictor model.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
import yaml
from loguru import logger
import argparse
import datetime
import torch
from torch.optim import AdamW
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Dataset, dataset, DataLoader


from models.downstream_vit import VisionTransformerForPredictor as vitpredictor
from models.downstream_vit import VisionTransformerForEncoder as vitencoder
from models.downstream_vit import TrainedEncoderPredictor
from models.downstream_vit import DownstreamHead


from utils import load_checkpoint_downstream, calculate_accuracy
from load_dataset import LoadLabelledDataset
import cred


DATETIME_NOW = datetime.datetime.now().replace(second=0, microsecond=0) #datetime without seconds & miliseconds.

def downstream(args):
    '''Main downstream module to retrain the predictor.
    '''

    
    with open(args.config, 'r') as configfile:
        config = yaml.load(configfile, Loader=yaml.FullLoader)
        print("Configuration file read successful...")

    with open(args.logging_config, 'r') as logging_configfile:
        logging_config = yaml.load(logging_configfile, Loader=yaml.FullLoader)
        print("Logging configuration file read successful...")
        print("Initializing logger...")

    if logging_config['disable_default_loggers']:
        logger.remove(0)

    logging_formatter = logging_config['formatters'][config['env']]

    
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


    #Model configurations
    PATCH_SIZE = config['model']['patch_size']
    TRAINED_ENCODER_PREDICTOR_SAVED_FOLDER = config['model']['trained_encoder_predictor_saved_folder']
    MODEL_SAVE_FOLDER = config['model']['model_save_folder']
    MODEL_NAME = config['model']['model_name']
    TRAINED_ENCODER_PREDICTOR_MODEL_NAME = config['model']['trained_encoder_predictor_model_name']
    MODEL_SAVE_FREQ = config['model']['model_save_freq']
    TRANSFORMER_DEPTH = config['model']['transformer_depth']
    ENCODER_NETWORK_EMBEDDING_DIM = config['model']['encoder_network_embedding_dim']
    PREDICTOR_NETWORK_EMBEDDING_DIM = config['model']['predictor_network_embedding_dim']
    PROJECTION_KEYS_DIM = config['model']['projection_keys_dim']
    PROJECTION_VALUES_DIM = config['model']['projection_values_dim']
    CLASSIFICATION_EMBEDDING_DIM = config['model']['classification_embedding_dim']
    FEEDFORWARD_PROJECTION_DIM = config['model']['feedforward_projection_dim']
    NUM_HEADS = config['model']['num_heads']
    ATTN_DROPOUT_PROB = config['model']['attn_dropout_prob']
    FEEDFORWARD_DROPOUT_PROB = config['model']['feedforward_dropout_prob']
    NUM_CLASS = config['model']['num_class']

    #Training configuration
    DEVICE = config['training']['device']
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and DEVICE=='gpu' else 'cpu')
    END_EPOCH = config['training']['end_epoch']
    START_EPOCH = config['training']['start_epoch']
    LEARNING_RATE = config['training']['learning_rate']
    USE_BFLOAT16 = config['training']['use_bfloat16']
    USE_NEPTUNE = config['training']['use_neptune']

    
    if USE_NEPTUNE:
        import neptune 

        NEPTUNE_RUN = neptune.init_run(
                                       project=cred.NEPTUNE_PROJECT,
                                       api_token=cred.NEPTUNE_API_TOKEN
                                       )
        
        NEPTUNE_RUN['parameters'] = neptune.utils.stringify_unsupported(config)

    NUM_PATCHES = (IMAGE_SIZE/PATCH_SIZE)**2 #for the initialization of param in predictor network.



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

    
    
    logger.info("Init Predictor model...")
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


    #REMEMBER WE DO NOT NEED THE TARGET ENCODER IN THIS MODULE.
    #Make sure the checkpoints are placed at the appropriate location.
    #We will load the model as is first before performing any changes for the downstream task.
    ENCODER_NETWORK, PREDICTOR_NETWORK = load_checkpoint_downstream(encoder_network=ENCODER_NETWORK,
                                                                    predictor_network=PREDICTOR_NETWORK,
                                                                    trained_model_folder=TRAINED_ENCODER_PREDICTOR_SAVED_FOLDER,
                                                                    trained_model_name=TRAINED_ENCODER_PREDICTOR_MODEL_NAME,
                                                                    load_checkpoint_epoch=None, 
                                                                    strict=True, 
                                                                    logger=logger)





    TRAINED_ENCODER_PREDICTOR_NETWORK = TrainedEncoderPredictor(trained_encoder=ENCODER_NETWORK,
                                                                trained_predictor=PREDICTOR_NETWORK,
                                                                num_patches=NUM_PATCHES,
                                                                predictor_network_embedding_dim=PREDICTOR_NETWORK_EMBEDDING_DIM,
                                                                device=DEVICE,
                                                                logger=logger)

    #set the mode to eval.
    TRAINED_ENCODER_PREDICTOR_NETWORK.eval()

    DOWNSTREAM_HEAD_NETWORK = DownstreamHead(predictor_network_embedding_dim=PREDICTOR_NETWORK_EMBEDDING_DIM,
                                             classification_embedding_dim=CLASSIFICATION_EMBEDDING_DIM,
                                             num_class=NUM_CLASS,
                                             device=DEVICE,
                                             logger=logger)


    summary(TRAINED_ENCODER_PREDICTOR_NETWORK, (IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE))

    summary(DOWNSTREAM_HEAD_NETWORK, (int(NUM_PATCHES), int(PREDICTOR_NETWORK_EMBEDDING_DIM)))


    #------------------------ Transforms settings.
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

    #------------------------------------------------



    #dataloader init.
    logger.info("Init dataset loading module...")
    TRAIN_DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER,
                                      transform=transforms.Compose(transforms_compose_list),
                                      logger=logger)

    TRAIN_DATASET_LOADER = DataLoader(TRAIN_DATASET_MODULE, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS) 

    TEST_DATASET_MODULE = LoadLabelledDataset(dataset_folder_path=DATASET_FOLDER,
                                      transform=transforms.ToTensor(), #REMEMBER, we do not need to augment test images. ToTensor automatically scales the images in the range of [0.0, 1.0].
                                      logger=logger,
                                      train=False)

    TEST_DATASET_LOADER = DataLoader(TEST_DATASET_MODULE, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS) 
    

    OPTIMIZER = AdamW(DOWNSTREAM_HEAD_NETWORK.parameters(), lr=LEARNING_RATE)
    CRITERION = torch.nn.CrossEntropyLoss().to(DEVICE)

    #scaler is used to scale the values in variables like state_dict, optimizer etc to bfloat16 type.
    if USE_BFLOAT16:
        SCALER = torch.cuda.amp.GradScaler()


    for epoch_idx in range(START_EPOCH, END_EPOCH):

        logger.info(f"Training has started for epoch {epoch_idx}")
        

        DOWNSTREAM_HEAD_NETWORK.train()
        train_epoch_loss = 0
        train_epoch_accuracy = []

        train_idx = None
        for train_idx, data in tqdm(enumerate(TRAIN_DATASET_LOADER)):

            #forward propagation.

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):

                batch_x, batch_y = data['image'].to(DEVICE), data['label'].to(DEVICE)


                feature_embedding = TRAINED_ENCODER_PREDICTOR_NETWORK(batch_x)
                prediction = DOWNSTREAM_HEAD_NETWORK(feature_embedding)

                batch_loss = CRITERION(input=prediction, target=batch_y)

                train_epoch_loss += batch_loss.item()

            #backward and step
            if USE_BFLOAT16:
                SCALER.scale(batch_loss).backward()
                SCALER.step(OPTIMIZER)
                SCALER.update()
            else:
                loss.backward()
                OPTIMIZER.step()

            OPTIMIZER.zero_grad()

            batch_accuracy = calculate_accuracy(predicted=prediction, target=batch_y)
            train_epoch_accuracy.append(batch_accuracy)

        train_epoch_accuracy = sum(train_epoch_accuracy)/(train_idx+1)
        logger.info(f"Train epoch loss at epoch {epoch_idx}: {train_epoch_loss}")
        logger.info(f"Train epoch accuracy at epoch {epoch_idx}: {train_epoch_accuracy}")


        logger.info(f"Testing has started for epoch {epoch_idx}")
        DOWNSTREAM_HEAD_NETWORK.eval()

        test_epoch_loss = 0
        test_epoch_accuracy = []

        test_idx = 0
        for test_idx, data in tqdm(enumerate(TEST_DATASET_LOADER)):

            #forward propagation.

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=USE_BFLOAT16):

                batch_x, batch_y = data['image'].to(DEVICE), data['label'].to(DEVICE)


                feature_embedding = TRAINED_ENCODER_PREDICTOR_NETWORK(batch_x)
                prediction = DOWNSTREAM_HEAD_NETWORK(feature_embedding)

                batch_loss = CRITERION(input=prediction, target=batch_y)


                test_epoch_loss += batch_loss.item()

                batch_accuracy = calculate_accuracy(predicted=prediction, target=batch_y)
                test_epoch_accuracy.append(batch_accuracy)



        test_epoch_accuracy = sum(test_epoch_accuracy)/(test_idx+1)
        logger.info(f"Test epoch loss at epoch {epoch_idx}: {test_epoch_loss}")
        logger.info(f"Test epoch accuracy at epoch {epoch_idx}: {test_epoch_accuracy}")

        if USE_NEPTUNE:
            NEPTUNE_RUN['train/loss_per_epoch'].append(train_epoch_loss)
            NEPTUNE_RUN['train/accuracy_per_epoch'].append(train_epoch_accuracy)
            NEPTUNE_RUN['test/loss_per_epoch'].append(test_epoch_loss)
            NEPTUNE_RUN['test/accuracy_per_epoch'].append(test_epoch_accuracy)




if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='Specify the YAML config file to be used.')
    parser.add_argument('--logging_config', required=True, type=str, help='Specify the YAML config file to be used for the logging module.')
    args = parser.parse_args()

    downstream(args)
