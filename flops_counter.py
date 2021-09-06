import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from thop import profile
from torch import nn
from args import cfg
import torch.nn.functional as F


def test_ptflops():
    net = models.resnet50()
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=False, print_per_layer_stat=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def test_thop():
    model = models.resnet50()
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<40}  {:<8}'.format('Number of parameters:', params))


def count_ptflops(model, inputs_dim, tag):
    macs, params = get_model_complexity_info(model, inputs_dim, as_strings=False, print_per_layer_stat=False, )
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


def count_thop(model, inputs_dim, tag):
    input = torch.randn((1,) + inputs_dim)
    macs, params = profile(model, inputs=(input,), verbose=False)
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


def count_thop2(model, i, tag):
    iss = torch.Tensor((1,) + i)
    macs, params = profile(model, inputs=(iss,), verbose=False)
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Computational complexity (MACs):', macs))
    print('{:<10} {:<40}  {:<8}'.format(tag.upper(), 'Number of parameters:', params))
    return macs, params


class ImgNetRS(nn.Module):
    def __init__(self, code_len, img_feat_len, hidden_len):
        super(ImgNetRS, self).__init__()
        # self.hash_layer = nn.Linear(img_feat_len, code_len)
        self.hash_net = nn.Sequential(
            nn.Linear(img_feat_len, hidden_len, bias=True),
            nn.BatchNorm1d(hidden_len),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_len, code_len, bias=True)
        )

        self.hash_net2 = nn.Sequential(
            nn.Linear(img_feat_len, hidden_len, bias=True),
            nn.BatchNorm1d(hidden_len),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_len, hidden_len, bias=True),
            nn.BatchNorm1d(hidden_len),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_len, code_len, bias=True)
        )

        self.hash_net2 = nn.Linear(img_feat_len, code_len, bias=True)

        self.alpha = 1.0

    def forward(self, x):
        x = F.normalize(x, dim=1)
        feat = F.normalize(x, dim=1)
        hid = self.hash_net(feat)

        code = torch.tanh(self.alpha * hid)

        return feat, hid, code


class TxtNetRS(nn.Module):
    def __init__(self, code_len, txt_feat_len, hidden_len):
        super(TxtNetRS, self).__init__()

        self.hash_net2 = nn.Sequential(
            nn.Linear(txt_feat_len, hidden_len, bias=True),
            nn.BatchNorm1d(hidden_len),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(hidden_len, code_len, bias=True)
        )

        self.hash_net = nn.Linear(txt_feat_len, code_len, bias=True)

        self.alpha = 1.0

    def forward(self, x):
        x = F.normalize(x, dim=1)
        hid = self.hash_net(x)
        code = torch.tanh(self.alpha * hid)
        return code


mi = ImgNetRS(cfg.HASH_BIT, 512, 4096)
mt = TxtNetRS(cfg.HASH_BIT, 768, 4096)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = mi


def calculate_stats_for_unhd(method='ptflops'):
    if method == 'ptflops':
        f = count_ptflops
    else:
        f = count_thop

    print('\n\n\n' + method + '\n')
    print('Module stats:')
    macsi, paramsi = f(mi, (512,), 'img')
    macst, paramst = f(mt, (768,), 'txt')

    total_params = paramsi + paramst
    total_macs = macsi * 2 + macst * 2

    print('\nTotal stats:')
    print('{:<40}  {:<8}'.format('Computational complexity (MACs):', total_macs))
    print('{:<40}  {:<8}'.format('Computational complexity (FLOPs):', total_macs * 2))
    print('{:<40}  {:<8}'.format('Number of parameters:', total_params))


def calculate_stats():
    calculate_stats_for_unhd()
    calculate_stats_for_unhd('thop')


if device.type == 'cpu':
    # test_ptflops()
    # test_thop()
    calculate_stats()
else:
    with torch.cuda.device(device):
        # test_ptflops()
        # test_thop()
        calculate_stats()
