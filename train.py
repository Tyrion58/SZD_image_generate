import argparse
from datetime import datetime
import logging
import os
import random
import time
import warnings
import wandb

from utils.logger import setlogger
from utils.train_utils import train_utils

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument("--model_name", default="cgan", type=str,
                        help="Model architecture")
    parser.add_argument('--data_name', type=str, default='SED_data', 
                        help='the name of the data')
    parser.add_argument("--data_path", default="./DATA", type=str,
                        help="Path to dataset.")
    parser.add_argument("--image_subpath", default="images", type=str, 
                        help="Path to dataset.")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', 
                        help='the directory to save the model')
    parser.add_argument("--epochs", default=30, type=int,
                        help="Number of total epochs to run. (Default: 30)")
    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="The batch size of the dataset. (Default: 32)")
    parser.add_argument("--lr", default=0.0002, type=float,
                        help="Learning rate. (Default: 0.0002)")
    parser.add_argument("--beta1", default=0.5, type=float,
                        help="beta1 of Adam. (Default: 0.5)")
    parser.add_argument("--image_size", default=64, type=int,
                        help="Image size. (Default: 64)")
    parser.add_argument("--channels", default=1, type=int,
                        help="The number of channels of the image. (Default: 1)")
    parser.add_argument("--netD", default="", type=str,
                        help="Path to Discriminator checkpoint.")
    parser.add_argument("--netG", default="", type=str,
                        help="Path to Generator checkpoint.")
    parser.add_argument("--pretrained", type=bool, default=True, 
                        help='whether to load the pretrained model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    sub_dir = args.model_name + '_' + args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

     # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    wandb.init(project="cgan_train")
    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()