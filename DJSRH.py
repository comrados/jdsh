import torch
import torch.nn.functional as F
import datasets
import os.path as osp
import os
from models import ImgNet, TxtNet, ImgNetRS, TxtNetRS
from utils import compress, calc_map_k, p_top_k, pr_curve, write_pickle
import time


class DJSRH:
    def __init__(self, log, cfg):
        self.since = time.time()
        self.logger = log
        self.cfg = cfg
        self.epoch_loss = 0.
        self.path = "_".join([self.cfg.MODEL, str(self.cfg.HASH_BIT), self.cfg.DATASET])

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(self.cfg.GPU_ID)

        if self.cfg.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True,
                                                       transform=datasets.mir_test_transform)

        if self.cfg.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        if self.cfg.DATASET == "UCM":
            self.train_dataset = datasets.UCM(train=True)
            self.test_dataset = datasets.UCM(train=False, database=False)
            self.database_dataset = datasets.UCM(train=False, database=True)

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.cfg.BATCH_SIZE,
                                                        shuffle=True,
                                                        num_workers=self.cfg.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=self.cfg.BATCH_SIZE,
                                                       shuffle=False,
                                                       num_workers=self.cfg.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=self.cfg.BATCH_SIZE,
                                                           shuffle=False,
                                                           num_workers=self.cfg.NUM_WORKERS)

        if self.cfg.DATASET == "UCM":
            txt_feat_len = datasets.txt_feat_len
            img_feat_len = datasets.img_feat_len

            self.ImgNet = ImgNetRS(self.cfg.HASH_BIT, img_feat_len, 512)
            self.TxtNet = TxtNetRS(self.cfg.HASH_BIT, txt_feat_len, 512)
        else:
            txt_feat_len = datasets.txt_feat_len
            self.ImgNet = ImgNet(code_len=self.cfg.HASH_BIT)
            self.TxtNet = TxtNet(code_len=self.cfg.HASH_BIT, txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD(self.ImgNet.parameters(), lr=self.cfg.LR_IMG, momentum=self.cfg.MOMENTUM,
                                     weight_decay=self.cfg.WEIGHT_DECAY)
        self.opt_T = torch.optim.SGD(self.TxtNet.parameters(), lr=self.cfg.LR_TXT, momentum=self.cfg.MOMENTUM,
                                     weight_decay=self.cfg.WEIGHT_DECAY)

        self.best_it = 0
        self.best_ti = 0
        self.best_ii = 0
        self.best_tt = 0

    def train(self, epoch):

        self.epoch_loss = 0.

        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)

        for idx, (img, txt, _, _) in enumerate(self.train_loader):

            img = torch.FloatTensor(img).cuda()
            txt = torch.FloatTensor(txt.numpy()).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            F_I, hid_I, code_I = self.ImgNet(img)
            code_T = self.TxtNet(txt)

            S = self.cal_similarity_matrix(F_I, txt)

            loss = self.cal_loss(code_I, code_T, S)

            self.epoch_loss += loss.detach().cpu().numpy()

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()

            if (idx + 1) % (len(self.train_dataset) // self.cfg.BATCH_SIZE / self.cfg.EPOCH_INTERVAL) == 0:
                # self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' % (epoch + 1, self.cfg.NUM_EPOCH, idx + 1, len(self.train_dataset) // self.cfg.BATCH_SIZE, loss.item()))
                pass

        self.logger.info('Epoch [%d/%d], Epoch Loss: %.4f' % (epoch + 1, self.cfg.NUM_EPOCH, self.epoch_loss))

    def eval(self):

        # self.logger.info('--------------------Evaluation: mAP@50-------------------')

        self.ImgNet.eval().cuda()
        self.TxtNet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.ImgNet,
                                                          self.TxtNet, self.database_dataset, self.test_dataset,
                                                          self.cfg.LABEL_DIM)

        MAP_I2T = calc_map_k(qu_BI, re_BT, qu_L, re_L)
        MAP_T2I = calc_map_k(qu_BT, re_BI, qu_L, re_L)

        MAP_I2I = calc_map_k(qu_BI, re_BI, qu_L, re_L)
        MAP_T2T = calc_map_k(qu_BT, re_BI, qu_L, re_L)

        MAPS = (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T)

        if (self.best_it + self.best_ti + self.best_ii + self.best_tt) < (MAP_I2T + MAP_T2I + MAP_I2I + MAP_T2T):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I
            self.best_ii = MAP_I2I
            self.best_tt = MAP_T2T

            if not self.cfg.TEST:
                self.save_checkpoints('best.pth')

        self.logger.info('mAP I->T: %.3f, mAP T->I: %.3f, mAP I->I: %.3f, mAP T->T: %.3f' % MAPS)
        # self.logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))
        # self.logger.info('--------------------------------------------------------------------')

    def test(self):
        self.ImgNet.eval().cuda()
        self.TxtNet.eval().cuda()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.ImgNet,
                                                          self.TxtNet, self.database_dataset, self.test_dataset,
                                                          self.cfg.LABEL_DIM)

        p_i2t, r_i2t = pr_curve(qu_BI, re_BT, qu_L, re_L, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qu_BT, re_BI, qu_L, re_L, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qu_BI, re_BI, qu_L, re_L, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qu_BT, re_BT, qu_L, re_L, tqdm_label='T2T')

        K = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
             10000]
        pk_i2t = p_top_k(qu_BI, re_BT, qu_L, re_L, K, tqdm_label='I2T')
        pk_t2i = p_top_k(qu_BT, re_BI, qu_L, re_L, K, tqdm_label='T2I')
        pk_i2i = p_top_k(qu_BI, re_BI, qu_L, re_L, K, tqdm_label='I2I')
        pk_t2t = p_top_k(qu_BT, re_BT, qu_L, re_L, K, tqdm_label='T2T')

        MAP_I2T = calc_map_k(qu_BI, re_BT, qu_L, re_L)
        MAP_T2I = calc_map_k(qu_BT, re_BI, qu_L, re_L)
        MAP_I2I = calc_map_k(qu_BI, re_BI, qu_L, re_L)
        MAP_T2T = calc_map_k(qu_BT, re_BI, qu_L, re_L)
        MAPS = (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T)

        pr_dict = {'pi2t': p_i2t.cpu().numpy(), 'ri2t': r_i2t.cpu().numpy(),
                   'pt2i': p_t2i.cpu().numpy(), 'rt2i': r_t2i.cpu().numpy(),
                   'pi2i': p_i2i.cpu().numpy(), 'ri2i': r_i2i.cpu().numpy(),
                   'pt2t': p_t2t.cpu().numpy(), 'rt2t': r_t2t.cpu().numpy()}

        pk_dict = {'k': K,
                   'pki2t': pk_i2t.cpu().numpy(),
                   'pkt2i': pk_t2i.cpu().numpy(),
                   'pki2i': pk_i2i.cpu().numpy(),
                   'pkt2t': pk_t2t.cpu().numpy()}

        map_dict = {'mapi2t': float(MAP_I2T.cpu().numpy()),
                    'mapt2i': float(MAP_T2I.cpu().numpy()),
                    'mapi2i': float(MAP_I2I.cpu().numpy()),
                    'mapt2t': float(MAP_T2T.cpu().numpy())}

        self.logger.info('mAP I->T: %.3f, mAP T->I: %.3f, mAP I->I: %.3f, mAP T->T: %.3f' % MAPS)

        write_pickle(osp.join(self.cfg.MODEL_DIR, self.path, 'pr_dict.pkl'), pr_dict)
        write_pickle(osp.join(self.cfg.MODEL_DIR, self.path, 'pk_dict.pkl'), pk_dict)
        write_pickle(osp.join(self.cfg.MODEL_DIR, self.path, 'map_dict.pkl'), map_dict)

    def cal_similarity_matrix(self, F_I, txt):

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(txt)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        S_tilde = self.cfg.BETA * S_I + (1 - self.cfg.BETA) * S_T
        S = (1 - self.cfg.ETA) * S_tilde + self.cfg.ETA * S_tilde.mm(S_tilde.t()) / self.cfg.BATCH_SIZE
        S = S * self.cfg.MU

        return S

    def cal_loss(self, code_I, code_T, S):

        B_I = F.normalize(code_I, dim=1)
        B_T = F.normalize(code_T, dim=1)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())

        loss1 = F.mse_loss(BI_BI, S)
        loss2 = F.mse_loss(BI_BT, S)
        loss3 = F.mse_loss(BT_BT, S)
        loss = self.cfg.LAMBDA1 * loss1 + 1 * loss2 + self.cfg.LAMBDA2 * loss3

        return loss

    def save_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.cfg.MODEL_DIR, self.path)
        if not os.path.exists(ckp_path):
            os.makedirs(ckp_path)
        ckp_path = osp.join(ckp_path, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully: ' + file_name + ' **********')

    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.cfg.MODEL_DIR, self.path, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError

            self.ImgNet.load_state_dict(obj['ImgNet'])
            self.TxtNet.load_state_dict(obj['TxtNet'])

    def training_coplete(self):
        MAPS = (self.best_it, self.best_ti, self.best_ii, self.best_tt)
        current = time.time()
        delta = current - self.since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))
        self.logger.info('Best mAPs: (I->T: %.3f, T->I: %.3f, I->I: %.3f, T->T: %.3f)' % MAPS)








