import numpy as np
import torch
import torch.nn.functional as F
import datasets
import os.path as osp
import os
from models import ImgNet, TxtNet, ImgNetRS, TxtNetRS
from utils import generate_hashes_from_dataloader, calc_map_k, p_top_k, pr_curve, write_pickle, calc_map_rad, build_binary_hists, top_k_hists, retrieval2png, hr_hists
import time

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class JointController:
    def __init__(self, log, cfg):
        self.since = time.time()
        self.logger = log
        self.cfg = cfg
        self.epoch_loss = 0.
        self.path = "_".join([self.cfg.MODEL, str(self.cfg.HASH_BIT), self.cfg.DATASET, self.cfg.TAG, self.cfg.DATA_AMOUNT.upper()])

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

        if self.cfg.DATASET == "UCM" or "RSICD":
            da = self.cfg.DATA_AMOUNT
            if da == 'aug':
                self.train_dataset = datasets.RSDSaug(type='train')
                self.test_dataset = datasets.RSDSaug(type='query')
                self.database_dataset = datasets.RSDSaug(type='db')
            else:
                self.train_dataset = datasets.RSDS(type='train', data_amount=da)
                self.test_dataset = datasets.RSDS(type='query')
                self.database_dataset = datasets.RSDS(type='db')

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

        if self.cfg.DATASET == "UCM" or "RSICD":
            txt_feat_len = datasets.txt_feat_len
            img_feat_len = datasets.img_feat_len

            self.ImgNet = ImgNetRS(self.cfg.HASH_BIT, img_feat_len, self.cfg.HIDDEN)
            self.TxtNet = TxtNetRS(self.cfg.HASH_BIT, txt_feat_len, self.cfg.HIDDEN)
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
        if self.cfg.MODEL == "DJSRH":
            self.train_djsrh(epoch)
        else:
            self.train_jdsh(epoch)

    def train_jdsh(self, epoch):

        self.epoch_loss = 0.

        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)

        S = None

        # plot distribution hist, get hyperparameters based on gaussian and laplacian distributions
        if epoch == 0:
            for idx, (img, txt, _, _, _) in enumerate(self.train_loader):

                with torch.no_grad():
                    img = torch.FloatTensor(img).cuda()
                    txt = torch.FloatTensor(txt.numpy()).cuda()

                    F_I, hid_I, code_I = self.ImgNet(img)
                    code_T = self.TxtNet(txt)

                    S_batch = self.get_S(F_I, txt)

                    if S is None:
                        S = S_batch
                    else:
                        S = torch.hstack([S, S_batch])

            S = S.detach().cpu().numpy()

            # remove high numbers (diagonal)
            S = S[S < 100]

            smin = S.min()
            smax = S.max()

            plt.figure(figsize=(20, 20))

            x = sns.displot(S, color='lightgray')

            # distribution
            Sm = x.ax.patches[np.argmax([h.get_height() for h in x.ax.patches])].xy[0]
            l = np.max([h.get_height() for h in x.ax.patches])
            plt.plot([Sm, Sm], [0, l * 1.1], c='red', label='Mode = {:2.3f}'.format(Sm))

            # gauss distr
            Sgauss_l = np.sort(S[S < Sm])
            Sgauss_r = Sm - Sgauss_l + Sm
            Sgauss = np.append(Sgauss_l, Sgauss_r, axis=0)  # array must be symmentric with regard to Sm
            mg, sg = stats.norm.fit(Sgauss)
            lsg = np.linspace(smin, Sm, len(Sgauss))
            pdf_g = stats.norm.pdf(lsg, mg, sg)
            pdf_g = pdf_g / pdf_g.max() * l
            plt.plot(lsg, pdf_g, label="Gaussian, m = {:2.3f}, s = {:2.3f}".format(mg, sg), c='orange')

            # laplace distr
            Slaplace_r = np.sort(S[S > Sm])
            Slaplace_l = Sm - Slaplace_r + Sm
            Slaplace = np.append(Slaplace_r, Slaplace_l, axis=0)  # array must be symmentric with regard to Sm
            ag, bg = stats.laplace.fit(Slaplace)
            lsl = np.linspace(Sm, smax, len(Slaplace))
            pdf_l = stats.laplace.pdf(lsl, ag, bg)
            pdf_l = pdf_l / pdf_l.max() * l
            plt.plot(lsl, pdf_l, label="Laplace, m = {:2.3f}, b = {:2.3f}".format(ag, bg), c='limegreen')

            plt.legend()
            plt.savefig(os.path.join('plots', 'dist_hist.png'))

        for idx, (img, txt, _, _, _) in enumerate(self.train_loader):

            img = torch.FloatTensor(img).cuda()
            txt = torch.FloatTensor(txt.numpy()).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            F_I, hid_I, code_I = self.ImgNet(img)
            code_T = self.TxtNet(txt)

            S = self.cal_similarity_matrix_jdsh(F_I, txt)

            loss = self.cal_loss_jdsh(code_I, code_T, S)

            self.epoch_loss += loss.detach().cpu().numpy()

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()

        self.logger.info('Epoch [%d/%d], Epoch Loss: %.4f' % (epoch + 1, self.cfg.NUM_EPOCH, self.epoch_loss))

    def get_S(self, F_I, txt):

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(txt)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        S_high = F.normalize(S_I).mm(F.normalize(S_T).t())
        S = self.cfg.alpha * S_I + self.cfg.beta * S_T + self.cfg.lamb * (S_high + S_high.t()) / 2

        """
        set diagonal (self-similarities) to high numbers
        this is a literally poor solution for removal of self-similarities (diagonal)
        self-similarities shall be in range [-alpha*beta*gamma, alpha*beta*gamma]
        just don't want to apply index-based removal, requires more thinking
        """
        y = torch.eye(len(S)).to(S.device) * 10000 + 1

        S = S * y

        return torch.reshape(S, (-1,))

    def train_djsrh(self, epoch):
        self.epoch_loss = 0.

        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)

        for idx, (img, txt, _, _, _) in enumerate(self.train_loader):
            img = torch.FloatTensor(img).cuda()
            txt = torch.FloatTensor(txt.numpy()).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()

            F_I, hid_I, code_I = self.ImgNet(img)
            code_T = self.TxtNet(txt)

            S = self.cal_similarity_matrix_djsrh(F_I, txt)

            loss = self.cal_loss_djsrh(code_I, code_T, S)

            self.epoch_loss += loss.detach().cpu().numpy()

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()

        self.logger.info('Epoch [%d/%d], Epoch Loss: %.4f' % (epoch + 1, self.cfg.NUM_EPOCH, self.epoch_loss))

    def cal_similarity_matrix_djsrh(self, F_I, txt):

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

    def cal_loss_djsrh(self, code_I, code_T, S):

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

    def eval(self):

        # self.logger.info('--------------------Evaluation: mAP@50-------------------')

        self.ImgNet.eval().cuda()
        self.TxtNet.eval().cuda()

        re_BI, re_BT, re_LT, qu_BI, qu_BT, qu_LT, indexes = generate_hashes_from_dataloader(self.database_loader,
                                                                                            self.test_loader,
                                                                                            self.ImgNet, self.TxtNet,
                                                                                            self.cfg.LABEL_DIM)

        # qu_BI = self.get_each_5th_element(qu_BI)
        # re_BI = self.get_each_5th_element(re_BI)
        qu_LI = qu_LT  # self.get_each_5th_element(qu_LT)
        re_LI = re_LT  # self.get_each_5th_element(re_LT)

        #indexes = list(indexes)
        #indexes[0] = self.get_each_5th_element(indexes[0])
        #indexes[2] = self.get_each_5th_element(indexes[2])



        MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, MAP_AVG = self.calc_maps_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI,
                                                                       re_LT, self.cfg.MAP_K)

        MAPS = (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T)

        maps5 = (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, MAP_AVG)
        maps10 = self.calc_maps_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, 10)
        maps20 = self.calc_maps_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, 20)
        mapshr = self.calc_maps_rad(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, [0, 1, 2, 3, 4, 5])
        p_at_k = self.calc_p_top_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT)


        if self.cfg.DRAW_PLOTS:
            self.visualize_retrieval(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, indexes, 'JDSH')
            top_k_hists(qu_BI, qu_BT, re_BI, re_BT, model='JDSH')
            hr_hists(qu_BI, qu_BT, re_BI, re_BT, model='JDSH')
            build_binary_hists(qu_BI, qu_BT, re_BI, re_BT, 'JDSH', [i[0] for i in mapshr])

        maps_eval = (maps5, maps10, maps20, mapshr, p_at_k)

        if (self.best_it + self.best_ti + self.best_ii + self.best_tt) < (MAP_I2T + MAP_T2I + MAP_I2I + MAP_T2T):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I
            self.best_ii = MAP_I2I
            self.best_tt = MAP_T2T

            if not self.cfg.TEST:
                self.save_checkpoints('best.pth')

        if not self.cfg.TEST:
            self.save_checkpoints('last.pth')
            write_pickle(osp.join(self.cfg.MODEL_DIR, self.path, 'maps_eval.pkl'), maps_eval)

        # self.logger.info('Best MAP of I->T: %.3f, Best mAP of T->I: %.3f' % (self.best_it, self.best_ti))
        # self.logger.info('--------------------------------------------------------------------')

    def visualize_retrieval(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, indexes, epoch):

        qIX, qIY, rIX, rIY = indexes

        cfg_params = [self.cfg.SEED, self.cfg.dataset_json_file, self.cfg.dataset_image_folder_path]

        i2t = retrieval2png(cfg_params, qBX, rBY, qLX, rLY, qIX, rIY, tag='I2T', file_tag=self.cfg.MODEL)
        t2i = retrieval2png(cfg_params, qBY, rBX, qLY, rLX, qIY, rIX, tag='T2I', file_tag=self.cfg.MODEL)
        i2i = retrieval2png(cfg_params, qBX, rBX, qLX, rLX, qIX, rIX, tag='I2I', file_tag=self.cfg.MODEL)
        t2t = retrieval2png(cfg_params, qBY, rBY, qLY, rLY, qIY, rIY, tag='T2T', file_tag=self.cfg.MODEL)

    @staticmethod
    def get_each_5th_element(arr):
        """
        intentionally ugly solution

        :return: array
        """
        return arr[::5]

    def test(self):
        self.ImgNet.eval().cuda()
        self.TxtNet.eval().cuda()

        re_BI, re_BT, re_LT, qu_BI, qu_BT, qu_LT, indexes = generate_hashes_from_dataloader(self.database_loader,
                                                                                            self.test_loader,
                                                                                            self.ImgNet, self.TxtNet,
                                                                                            self.cfg.LABEL_DIM)

        # qu_BI = self.get_each_5th_element(qu_BI)
        # re_BI = self.get_each_5th_element(re_BI)
        qu_LI = qu_LT  # self.get_each_5th_element(qu_LT)
        re_LI = re_LT  # self.get_each_5th_element(re_LT)

        # indexes = list(indexes)
        # indexes[0] = self.get_each_5th_element(indexes[0])
        # indexes[2] = self.get_each_5th_element(indexes[2])

        MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, MAP_AVG = self.calc_maps_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI,
                                                                       re_LT, self.cfg.MAP_K)

        MAPS = (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T)

        maps5 = (MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, MAP_AVG)
        maps10 = self.calc_maps_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, 10)
        maps20 = self.calc_maps_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, 20)
        mapshr = self.calc_maps_rad(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, [0, 1, 2, 3, 4, 5])
        p_at_k = self.calc_p_top_k(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT)

        self.visualize_retrieval(qu_BI, qu_BT, re_BI, re_BT, qu_LI, qu_LT, re_LI, re_LT, indexes, 'JDSH')

        p_i2t, r_i2t = pr_curve(qu_BI, re_BT, qu_LI, re_LT, tqdm_label='I2T')
        p_t2i, r_t2i = pr_curve(qu_BT, re_BI, qu_LT, re_LI, tqdm_label='T2I')
        p_i2i, r_i2i = pr_curve(qu_BI, re_BI, qu_LI, re_LI, tqdm_label='I2I')
        p_t2t, r_t2t = pr_curve(qu_BT, re_BT, qu_LT, re_LT, tqdm_label='T2T')

        K = [1, 10, 50] + list(range(100, 1000, 100)) + list(range(1000, 10001, 1000))
        pk_i2t = p_top_k(qu_BI, re_BT, qu_LI, re_LT, K, tqdm_label='I2T')
        pk_t2i = p_top_k(qu_BT, re_BI, qu_LT, re_LI, K, tqdm_label='T2I')
        pk_i2i = p_top_k(qu_BI, re_BI, qu_LI, re_LI, K, tqdm_label='I2I')
        pk_t2t = p_top_k(qu_BT, re_BT, qu_LT, re_LT, K, tqdm_label='T2T')

        MAP_I2T = calc_map_k(qu_BI, re_BT, qu_LI, re_LT, self.cfg.MAP_K)
        MAP_T2I = calc_map_k(qu_BT, re_BI, qu_LT, re_LI, self.cfg.MAP_K)
        MAP_I2I = calc_map_k(qu_BI, re_BI, qu_LI, re_LI, self.cfg.MAP_K)
        MAP_T2T = calc_map_k(qu_BT, re_BT, qu_LT, re_LT, self.cfg.MAP_K)
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

    def cal_similarity_matrix_jdsh(self, F_I, txt):

        F_I = F.normalize(F_I)
        S_I = F_I.mm(F_I.t())
        S_I = S_I * 2 - 1

        F_T = F.normalize(txt)
        S_T = F_T.mm(F_T.t())
        S_T = S_T * 2 - 1

        S_high = F.normalize(S_I).mm(F.normalize(S_T).t())
        S_ = self.cfg.alpha * S_I + self.cfg.beta * S_T + self.cfg.lamb * (S_high + S_high.t()) / 2

        #         S_ones = torch.ones_like(S_).cuda()
        #         S_eye = torch.eye(S_.size(0), S_.size(1)).cuda()
        #         S_mask = S_ones - S_eye

        left = self.cfg.LOC_LEFT - self.cfg.ALPHA * self.cfg.SCALE_LEFT
        right = self.cfg.LOC_RIGHT + self.cfg.BETA * self.cfg.SCALE_RIGHT

        #S_[S_ < left] = (1 + self.cfg.L1 * torch.exp(-(S_[S_ < left] - self.cfg.MIN))) * S_[S_ < left]
        #S_[S_ > right] = (1 + self.cfg.L2 * torch.exp(S_[S_ > right] - self.cfg.MAX)) * S_[S_ > right]

        S = S_ * self.cfg.mu

        return S

    def cal_loss_jdsh(self, code_I, code_T, S):

        B_I = F.normalize(code_I, dim=1)
        B_T = F.normalize(code_T, dim=1)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        BT_BI = B_T.mm(B_I.t())

        loss1 = F.mse_loss(BI_BI, S)
        loss2 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) - (B_I * B_T).sum(dim=1).mean()
        loss3 = F.mse_loss(BT_BT, S)

        loss = self.cfg.INTRA * loss1 + loss2 + self.cfg.INTRA * loss3

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

    def training_complete(self):
        MAPS = (self.best_it, self.best_ti, self.best_ii, self.best_tt)
        current = time.time()
        delta = current - self.since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(delta // 60, delta % 60))
        self.logger.info('Best mAPs: (I->T: %.3f, T->I: %.3f, I->I: %.3f, T->T: %.3f)' % MAPS)

    def calc_maps_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, k):
        """
        Calculate MAPs, in regards to K

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: k: k

        :returns: MAPs
        """
        mapi2t = calc_map_k(qBX, rBY, qLX, rLY, k)
        mapt2i = calc_map_k(qBY, rBX, qLY, rLX, k)
        mapi2i = calc_map_k(qBX, rBX, qLX, rLX, k)
        mapt2t = calc_map_k(qBY, rBY, qLY, rLY, k)

        avg = (mapi2t.item() + mapt2i.item() + mapi2i.item() + mapt2t.item()) * 0.25

        mapi2t, mapt2i, mapi2i, mapt2t, mapavg = mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), avg

        s = 'Valid: mAP@{}, avg: {:3.3f}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        self.logger.info(s.format(k, mapavg, mapi2t, mapt2i, mapi2i, mapt2t))

        return mapi2t, mapt2i, mapi2i, mapt2t, mapavg

    def calc_maps_rad(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY, rs):
        """
        Calculate MAPs, in regard to Hamming radius

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y
        :param: rs: hamming radiuses to output

        :returns: MAPs
        """
        mapsi2t = calc_map_rad(qBX, rBY, qLX, rLY)
        mapst2i = calc_map_rad(qBY, rBX, qLY, rLX)
        mapsi2i = calc_map_rad(qBX, rBX, qLX, rLX)
        mapst2t = calc_map_rad(qBY, rBY, qLY, rLY)

        mapsi2t, mapst2i, mapsi2i, mapst2t = mapsi2t.numpy(), mapst2i.numpy(), mapsi2i.numpy(), mapst2t.numpy()

        s = 'Valid: mAP HR{}, i->t: {:3.3f}, t->i: {:3.3f}, i->i: {:3.3f}, t->t: {:3.3f}'
        for r in rs:
            self.logger.info(s.format(r, mapsi2t[r], mapst2i[r], mapsi2i[r], mapst2t[r]))

        return mapsi2t, mapst2i, mapsi2i, mapst2t

    def calc_p_top_k(self, qBX, qBY, rBX, rBY, qLX, qLY, rLX, rLY):
        """
        Calculate P@K values

        :param: qBX: query hashes, modality X
        :param: qBY: query hashes, modality Y
        :param: rBX: response hashes, modality X
        :param: rBY: response hashes, modality Y
        :param: qLX: query labels, modality X
        :param: qLY: query labels, modality Y
        :param: rLX: response labels, modality X
        :param: rLY: response labels, modality Y

        :returns: P@K values
        """
        k = [1, 5, 10, 20, 50] + list(range(100, 1001, 100))

        pk_i2t = p_top_k(qBX, rBY, qLX, rLY, k, tqdm_label='I2T').detach().cpu().numpy()
        pk_t2i = p_top_k(qBY, rBX, qLY, rLX, k, tqdm_label='T2I').detach().cpu().numpy()
        pk_i2i = p_top_k(qBX, rBX, qLX, rLX, k, tqdm_label='I2I').detach().cpu().numpy()
        pk_t2t = p_top_k(qBY, rBY, qLY, rLY, k, tqdm_label='T2T').detach().cpu().numpy()

        pk_dict = {'k': k,
                   'pki2t': list(pk_i2t),
                   'pkt2i': list(pk_t2i),
                   'pki2i': list(pk_i2i),
                   'pkt2t': list(pk_t2t)}

        self.logger.info('P@K values: {}'.format(pk_dict))

        return pk_dict