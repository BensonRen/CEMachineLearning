#Parameter.py where I put all the parameters
import argparse
DEVICE = 'cuda:1'
RESULT_DIR = 'results'
# Setting some hyperparameters
TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 100
INITIAL_LR = 0.01
MOMENTUM = 0.9
REG = 5e-4
EPOCHS = 1
DATAROOT = "./data"
CHECKPOINT_PATH = "./saved_model"
TRAIN_FROM_SCRATCH = False
DECAY_EPOCH = 1
DECAY_RATE  = 0.98
