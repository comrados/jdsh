import argparse
from easydict import EasyDict as edict
import json

parser = argparse.ArgumentParser(description='JDSH and DJSRH')
parser.add_argument('--Train', default=True, help='train or test', type=bool)
parser.add_argument('--Dataset', default='UCM', help='MIRFlickr or NUSWIDE', type=str)
parser.add_argument('--Checkpoint', default='UCM.pth', help='checkpoint name', type=str)
parser.add_argument('--Bit', default=128, help='hash bit', type=int)
parser.add_argument('--Model', default='JDSH', help='JDSH or DJSRH', type=str)

args = parser.parse_args()

config_file = r"./config/" + args.Model + r"_" + args.Dataset + r".json"

# load basic settings
with open(config_file, 'r') as f:
    config = edict(json.load(f))

# update settings
config.TRAIN = args.Train
config.DATASET = args.Dataset
config.CHECKPOINT = args.Checkpoint
config.HASH_BIT = args.Bit
config.MODEL = args.Model
