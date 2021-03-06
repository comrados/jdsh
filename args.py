import argparse
import json

parser = argparse.ArgumentParser(description='JDSH and DJSRH')
parser.add_argument('--test', default=False, help='train or test', action='store_true')
parser.add_argument('--dataset', default='UCM', help='UCM or RSICD', type=str)
parser.add_argument('--checkpoint', default='UCM.pth', help='checkpoint name', type=str)
parser.add_argument('--bit', default=128, help='hash bit', type=int)
parser.add_argument('--model', default='JDSH', help='JDSH or DJSRH', type=str)
parser.add_argument('--tag', default='default', help='model tag', type=str)
parser.add_argument('--data-amount', default='normal', help="data amount: 'normal', 'double', 'aug'", type=str)
parser.add_argument('--plots', default=False, help="build plots or not", action='store_true')
parser.add_argument('--hidden-layer', default=4096, help="hidden layer size", type=int)

args = parser.parse_args()

config_file = r"./config/" + args.model + r"_" + args.dataset.upper() + r".json"

# load basic settings
with open(config_file, 'r') as f:
    cfg = json.load(f)


class Config:

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Config, self).__setattr__(name, value)


cfg = Config(cfg)

# update settings
cfg.TEST = args.test
cfg.HIDDEN = args.hidden_layer
cfg.DATASET = args.dataset.upper()
cfg.CHECKPOINT = args.checkpoint
cfg.HASH_BIT = args.bit
cfg.MODEL = args.model
cfg.TAG = args.tag.upper()
cfg.DATA_AMOUNT = args.data_amount
cfg.DRAW_PLOTS = args.plots
