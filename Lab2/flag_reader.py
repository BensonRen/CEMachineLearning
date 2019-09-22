import argparse
import pandas as pd
import os
from parameters import *

def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', type=str, default=RESULT_DIR,
                        help='The dir to put result')
    parser.add_argument('--device', type=str, default=DEVICE,
                        help='The device that you would like to work on')
    parser.add_argument('--train-batch-size', type=int, default=TRAIN_BATCH_SIZE,
                        help='batch size of your training data')
    parser.add_argument('--val-batch-size', type=int, default=VAL_BATCH_SIZE,
                        help='batch size of your validation data')
    parser.add_argument('--initial-lr', type=float, default=INITIAL_LR,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help='momentum of your optimizer')
    parser.add_argument('--reg', type=float, default=REG,
                        help='regularization constant')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='#epochs')
    parser.add_argument('--decay-epochs', type=int, default=DECAY_EPOCH,
                        help='#epochs to decay')
    parser.add_argument('--decay-rate', type=float, default=DECAY_RATE,
                        help='rate to decay')
    parser.add_argument('--dataroot', type=str, default=DATAROOT,
                        help='the place where you put your data')
    parser.add_argument('--checkpoint-path', type=str, default=CHECKPOINT_PATH,
                        help='the path you place your checkpoint')
    parser.add_argument('--train-from-scratch', type=bool, default=TRAIN_FROM_SCRATCH,
                        help='the flag that whether you train it from scratch')
    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code

    # flagsVar = vars(flags)

    return flags

def write_flags(flags, best_validation_loss, dirname):
    # To avoid terrible looking shape of y_range
    flags_dict = vars(flags)
    flags_dict_copy = flags_dict.copy()  # in order to not corrupt the original data strucutre
    flags_dict_copy['best_validation_loss'] = best_validation_loss
    # Convert the dictionary into pandas data frame which is easier to handle with and write read
    print(flags_dict_copy)
    flags_df = pd.DataFrame.from_dict(flags_dict_copy, orient='index', columns=['value'])
    flags_df_transpose = flags_df.transpose()
    flags_df_transpose.to_csv(os.path.join(dirname,"parameters.txt"))
