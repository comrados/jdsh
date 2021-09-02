import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ImgNetRS(nn.Module):
    def __init__(self, code_len, img_feat_len, hidden_len):
        super(ImgNetRS, self).__init__()

        # paper's logic: if feature encoding net is not trainable: features -> hidden FC (4096) -> hash code
        self.hash_net = nn.Sequential(
            nn.Linear(img_feat_len, hidden_len, bias=True),
            nn.BatchNorm1d(hidden_len),
            nn.ReLU(True),
            nn.Linear(hidden_len, code_len, bias=True)
        )

        # paper's logic: if feature encoding net is trainable: features -> hash code
        self.hash_net_ = nn.Linear(img_feat_len, code_len, bias=True)

        self.alpha = 1.0

    def forward(self, x):
        x = F.normalize(x, dim=1)
        feat = F.normalize(x, dim=1)
        hid = self.hash_net(feat)

        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNetRS(nn.Module):
    def __init__(self, code_len, txt_feat_len, hidden_len):
        super(TxtNetRS, self).__init__()

        # paper's logic: if feature encoding net is not trainable: features -> hidden FC (4096) -> hash code
        self.hash_net = nn.Sequential(
            nn.Linear(txt_feat_len, hidden_len, bias=True),
            nn.BatchNorm1d(hidden_len),
            nn.ReLU(True),
            nn.Linear(hidden_len, code_len, bias=True)
        )

        # paper's logic: if feature encoding net is trainable: features -> hash code
        self.hash_net_ = nn.Linear(txt_feat_len, code_len, bias=True)

        self.alpha = 1.0

    def forward(self, x):
        x = F.normalize(x, dim=1)
        hid = self.hash_net(x)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()

        # trainable feature encoding net
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.hash_layer = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        with torch.no_grad():
            x = self.alexnet.features(x)
            x = x.view(x.size(0), -1)
            feat = self.alexnet.classifier(x)

        hid = self.hash_layer(feat)
        feat = F.normalize(feat, dim=1)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()

        # non-trainable feature encoding net
        self.net = nn.Sequential(nn.Linear(txt_feat_len, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4096, code_len),
                                 )

        self.alpha = 1.0

    def forward(self, x):
        hid = self.net(x)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)