import argparse
import logging
import os
import random
import warnings

import torch

from utils.train_utils import train_utils

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--model_name", default="cgan", type=str,
                        help="Model architecture")
    parser.add_argument("--model_dir", type=str,
                        help="Model architecture")
    parser.add_argument("--save_dir", default="./eval_images", type=str,
                        help="Path to dataset.")
    parser.add_argument('--data_name', type=str, default='SED_data', 
                        help='the name of the data')
    parser.add_argument("--data_path", default="./DATA", type=str,
                        help="Path to dataset.")
    parser.add_argument("--image_subpath", default="images", type=str, 
                        help="Path to dataset.")
    parser.add_argument("--train", type=bool, default=False, 
                        help='whether to load the pretrained model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    
    trainer = train_utils(args, args.save_dir)
    trainer.setup()
    trainer.evaluate()