import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
import os.path as osp
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os
from args import cfg
import random
from PIL import Image
import json
import h5py


def read_hdf5(file_name, dataset_name, normalize=False):
    with h5py.File(file_name, 'r') as hf:
        print("Read from:", file_name)
        data = hf[dataset_name][:]
        if normalize:
            data = (data - data.mean()) / data.std()
        return data


def get_labels(data, suppress_console_info=False):
    labels = []
    for img in data['images']:
        labels.append(img["classcode"])
    if not suppress_console_info:
        print("Total number of labels:", len(labels))
    return labels


def get_captions(data, suppress_console_info=False):

    def format_caption(string):
        return string.replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower()

    captions = []
    augmented_captions_rb = []
    augmented_captions_bt_prob = []
    augmented_captions_bt_chain = []
    for img in data['images']:
        for sent in img['sentences']:
            captions.append(format_caption(sent['raw']))
            try:
                augmented_captions_rb.append(format_caption(sent['aug_rb']))
            except:
                pass
            try:
                augmented_captions_bt_prob.append(format_caption(sent['aug_bt_prob']))
            except:
                pass
            try:
                augmented_captions_bt_chain.append(format_caption(sent['aug_bt_chain']))
            except:
                pass
    if not suppress_console_info:
        print("Total number of captions:", len(captions))
        print("Total number of augmented captions RB:", len(augmented_captions_rb))
        print("Total number of augmented captions BT (prob):", len(augmented_captions_bt_prob))
        print("Total number of augmented captions BT (chain):", len(augmented_captions_bt_chain))
    return captions, augmented_captions_rb, augmented_captions_bt_prob, augmented_captions_bt_chain


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def p_topK(qB, rB, query_label, retrieval_label, K=None):
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target)

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, _, code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, data_T, target, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target)

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, _, code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def generate_hashes_from_dataloader(db_loader, q_loader, model_I, model_T, label_dim):
    def stack_idxs(idxs, idxs_batch):
        if len(idxs) == 0:
            return [ib for ib in idxs_batch]
        else:
            return [torch.hstack(i).detach() for i in zip(idxs, idxs_batch)]

    # response hashes / labels
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    db_dl_idxs = []
    for _, (data_I, data_T, lab, _, db_sample_idxs) in enumerate(db_loader):
        db_dl_idxs = stack_idxs(db_dl_idxs, db_sample_idxs)
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

        re_L.extend(lab.cpu().data.numpy())

    # query hashes / labels
    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    q_dl_idxs = []
    for _, (data_I, data_T, lab, _, q_sample_idxs) in enumerate(q_loader):
        q_dl_idxs = stack_idxs(q_dl_idxs, q_sample_idxs)
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        code_T = model_T(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

        qu_L.extend(lab.cpu().data.numpy())

    # prepare to output
    re_BI = torch.from_numpy(np.array(re_BI)).cuda()
    re_BT = torch.from_numpy(np.array(re_BT)).cuda()
    re_L = torch.from_numpy(np.array(re_L)).cuda()
    if len(re_L.shape) == 1:
        re_L = F.one_hot(re_L, num_classes=label_dim).float().cuda()

    qu_BI = torch.from_numpy(np.array(qu_BI)).cuda()
    qu_BT = torch.from_numpy(np.array(qu_BT)).cuda()
    qu_L = torch.from_numpy(np.array(qu_L)).cuda()
    if len(qu_L.shape) == 1:
        qu_L = F.one_hot(qu_L, num_classes=label_dim).float().cuda()

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L, (q_dl_idxs[0], q_dl_idxs[1], db_dl_idxs[0], db_dl_idxs[1])


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def logger():
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    log_name = 'log.txt'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)

    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)

    return logger


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    """
    calculate MAPs

    :param qB: query binary codes
    :param rB: response binary codes
    :param query_label: labels of query
    :param retrieval_label: labels of response
    :param k: k
    :return:
    """
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_map_rad(qB, rB, query_label, retrieval_label):
    """
    calculate MAPs, in regard to hamming radius

    :param qB: query binary codes
    :param rB: response binary codes
    :param query_label: labels of query
    :param retrieval_label: labels of response
    :return:
    """

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)

    # for each sample from query calculate precision and recall
    for i in range(num_query):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)  # hamming distances from qB[i, :] (current sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from query sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        P[i] = p
    P = P.mean(dim=0)
    return P


def pr_curve(qB, rB, query_label, retrieval_label, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'PR-curve ' + tqdm_label

    num_query = qB.shape[0]  # length of query (each sample from query compared to retrieval samples)
    num_bit = qB.shape[1]  # length of hash code
    P = torch.zeros(num_query, num_bit + 1)  # precisions (for each sample)
    R = torch.zeros(num_query, num_bit + 1)  # recalls (for each sample)

    # for each sample from query calculate precision and recall
    for i in tqdm(range(num_query), desc=tqdm_label):
        # gnd[j] == 1 if same class, otherwise 0, ground truth
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # tsum (TP + FN): total number of samples of the same class
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :],
                                 rB)  # hamming distances from qB[i, :] (current query sample) to retrieval samples
        # tmp[k,j] == 1 if hamming distance to retrieval sample j is less or equal to k (distance), 0 otherwise
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        # total (TP + FP): total[k] is count of distances less or equal to k (from current sample to retrieval samples)
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.0001  # replace zeros with 0.1 to avoid division by zero
        # select only same class samples from tmp (ground truth masking, only rows where gnd == 1 proceed further)
        t = gnd * tmp
        # count (TP): number of true (correctly selected) samples of the same class for any given distance k
        count = t.sum(dim=-1)
        p = count / total  # TP / (TP + FP)
        r = count / tsum  # TP / (TP + FN)
        P[i] = p
        R[i] = r
    # mask to calculate P mean value (among all query samples)
    # mask = (P > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # P = P.sum(dim=0) / mask
    # mask to calculate R mean value (among all query samples)
    # mask = (R > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.001
    # R = R.sum(dim=0) / mask
    P = P.mean(dim=0)
    R = R.mean(dim=0)
    return P, R


def p_top_k(qB, rB, query_label, retrieval_label, K, tqdm_label=''):
    if tqdm_label != '':
        tqdm_label = 'AP@K ' + tqdm_label

    num_query = qB.shape[0]
    PK = torch.zeros(len(K)).to(qB.device)

    for i in tqdm(range(num_query), desc=tqdm_label):
        # ground_truth[j] == 1 if same class (if at least 1 same label), otherwise 0, ground truth
        ground_truth = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        # count of samples, that shall be retrieved
        tp_fn = ground_truth.sum()
        if tp_fn == 0:
            continue

        hamm_dist = calc_hamming_dist(qB[i, :], rB).squeeze()

        # for each k in K
        for j, k in enumerate(K):
            k = min(k, retrieval_label.shape[0])
            _, sorted_indexes = torch.sort(hamm_dist)
            retrieved_indexes = sorted_indexes[:k]
            retrieved_samples = ground_truth[retrieved_indexes]
            PK[j] += retrieved_samples.sum() / k

    PK = PK / num_query

    return PK


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def build_binary_hists(qBX, qBY, rBX, rBY, model, maps_r0):
    def get_dict_of_binaries(binary_codes):

        def bin2dec(bin):
            bin = bin.detach().cpu().numpy()
            dec = np.uint64(0)
            for mag, bit in enumerate(bin[::-1]):
                dec += np.uint64(1 if bit >= 0 else 0) * np.power(np.uint64(2), np.uint64(mag), dtype=np.uint64)
            return dec

        dict_of_binaries = {}
        l = binary_codes.shape[0]
        for i in range(l):
            dec = bin2dec(binary_codes[i])
            if dec not in dict_of_binaries:
                dict_of_binaries[dec] = 1
            else:
                dict_of_binaries[dec] += 1

        return dict_of_binaries

    def get_stacked_bar_dict(qBd, rBd):
        joint_dict = qBd.copy()
        for k, v in joint_dict.items():
            joint_dict[k] = (v, 0)
        for k, v in rBd.items():
            if k in joint_dict:
                joint_dict[k] = (joint_dict[k][0], v)
            else:
                joint_dict[k] = (0, v)
        return joint_dict

    def plot_stacked_bar(stacked_bar_dict, tag, ax):
        labels = [str(i) for i in stacked_bar_dict.keys()]
        q = [i[0] for i in stacked_bar_dict.values()]
        r = [i[1] for i in stacked_bar_dict.values()]
        width = 1

        ax.bar(labels, q, width, label='Query')
        ax.bar(labels, r, width, bottom=q, label='Recall')
        plt.xticks([])
        plt.grid()
        ax.set_ylabel('Quantity')
        ax.set_title(tag.upper(), size=50, weight='medium')
        ax.legend()
        plt.tight_layout()

    qBXd = get_dict_of_binaries(qBX)
    qBYd = get_dict_of_binaries(qBY)
    rBXd = get_dict_of_binaries(rBX)
    rBYd = get_dict_of_binaries(rBY)

    i2t = get_stacked_bar_dict(qBXd, rBYd)
    t2i = get_stacked_bar_dict(qBYd, rBXd)
    i2i = get_stacked_bar_dict(qBXd, rBXd)
    t2t = get_stacked_bar_dict(qBYd, rBYd)

    tags = ['i2t', 't2i', 'i2i', 't2t']
    dicts = [i2t, t2i, i2i, t2t]

    fig = plt.figure(figsize=(60, 40))
    for i, (tag, d, mr0) in enumerate(zip(tags, dicts, maps_r0)):
        bins_used = 'buckets: {} / {}'.format(len(d), 2 ** cfg.HASH_BIT)
        experiment = ', '.join([tag, model, "mAP HR0: {:3.3f}".format(mr0), bins_used])
        ax = fig.add_subplot(2, 2, i + 1)
        plot_stacked_bar(d, experiment, ax)
    plt.savefig(os.path.join('plots', 'hists_' + model + '.png'))


def top_k_hists(qBX, qBY, rBX, rBY, k=10, model=''):
    def top_k_hist_data(qB, rB, k):
        n = len(qB)
        d = {}
        for i in range(n):
            ham_dist = calc_hamming_dist(qB[i, :], rB).squeeze().detach().cpu()
            ham_dist_sorted, idxs = torch.sort(ham_dist)

            ham_dist_sorted_k = ham_dist_sorted[:k].cpu().numpy()
            values, counts = np.unique(ham_dist_sorted_k, return_counts=True)
            for v, c in zip(values.astype(int), counts):
                if v in d:
                    d[v] += c
                else:
                    d[v] = c

        x = list(range(max(d.keys())))
        y = [d[j] if j in d else 0 for j in range(max(d.keys()))]

        return x, y, n

    def plot_top_k_hist(x, y, n, tag, ax):
        rects = ax.bar(x, y, width=1)
        try:
            scale = max([rect.get_height() for rect in rects])
        except:
            scale = 100
        for rect in rects:
            h = rect.get_height()
            if h < scale * 0.1:
                txt_offset = int(scale * 0.05)
            else:
                txt_offset = - int(scale * 0.05)
            ax.annotate('Mean: {:.1f}'.format(h / n), xy=(rect.get_x() + rect.get_width() / 2 - 0.25, h + txt_offset),
                        weight='bold', color='red')
        plt.title(tag.upper(), size=20, weight='medium')
        plt.grid(axis='y')
        ax.set_xticks(x)

    i2t = top_k_hist_data(qBX, rBY, k)
    t2i = top_k_hist_data(qBY, rBX, k)
    i2i = top_k_hist_data(qBX, rBX, k)
    t2t = top_k_hist_data(qBY, rBY, k)

    data = [i2t, t2i, i2i, t2t]
    tags = ['i2t', 't2i', 'i2i', 't2t']

    fig = plt.figure(figsize=(16, 8))
    for i, (tag, d) in enumerate(zip(tags, data)):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_top_k_hist(*d, ', '.join([tag, model, 'k: {}'.format(k), 'q/r: {}/{}'.format(d[-1], d[-1] * k)]), ax)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'top_k_hists_' + model + '.png'))


def read_json(file_name, suppress_console_info=False):
    """
    Read JSON

    :param file_name: input JSON path
    :param suppress_console_info: toggle console printing
    :return: dictionary from JSON
    """
    with open(file_name, 'r') as f:
        data = json.load(f)
        if not suppress_console_info:
            print("Read from:", file_name)
    return data


def get_image_file_names(data, suppress_console_info=False):
    """
    Get list of image file names

    :param data: original data from JSON
    :param suppress_console_info: toggle console printing
    :return: list of strings (file names)
    """

    file_names = []
    for img in data['images']:
        file_names.append(img["filename"])
    if not suppress_console_info:
        print("Total number of files:", len(file_names))
    return file_names


def get_captions(data, suppress_console_info=False):
    """
    Get list of formatted captions

    :param data: original data from JSON
    :return: list of strings (captions)
    """

    def format_caption(string):
        return string.replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower()

    captions = []
    augmented_captions = []
    for img in data['images']:
        for sent in img['sentences']:
            captions.append(format_caption(sent['raw']))
            try:
                augmented_captions.append(format_caption(sent['aug']))
            except:
                continue
    if not suppress_console_info:
        print("Total number of captions:", len(captions))
        print("Total number of augmented captions:", len(augmented_captions))
    return captions, augmented_captions


def retrieval2png(cfg_params, qB, rB, qL, rL, qI, rI, k=20, tag='', file_tag='UNHD'):
    print('Visualizing retrieval for:', tag)

    def get_retrieved_info(qB, rB, qL, rL, qI, rI, k):

        i = 50

        ham_dist = calc_hamming_dist(qB[i, :], rB).squeeze().detach().cpu()
        ham_dist_sorted, idxs = torch.sort(ham_dist)

        ham_dist_sorted_k = ham_dist_sorted[:k].cpu().numpy()
        idxs_k = idxs[:k].cpu().numpy()

        q_idx = qI[i].cpu().numpy()
        r_idxs = rI[idxs_k].cpu().numpy()
        q_lab = np.argmax(qL[i].cpu().numpy())
        r_labs = np.argmax(rL[idxs_k].cpu().numpy(), axis=1)

        return ham_dist_sorted_k, q_idx, r_idxs, q_lab, r_labs

    def load_img_txt():
        data = read_json(cfg_params[1], True)
        file_names = get_image_file_names(data, True)
        img_paths = [os.path.join(cfg_params[2], i) for i in file_names]
        captions, _ = get_captions(data, True)
        return img_paths, captions

    def get_retrieval_dict(img_paths, captions, q_idx, r_idxs, tag):
        d = {'tag': tag}
        if tag.startswith('I'):
            d['q'] = img_paths[q_idx]
            d['o'] = captions[q_idx]  # captions[q_idx*5]
        else:
            d['q'] = captions[q_idx]  # captions[q_idx*5]
            d['o'] = img_paths[q_idx]
        if tag.endswith('I'):
            d['r'] = [img_paths[r_idx] for r_idx in r_idxs]
        else:
            d['r'] = [captions[r_idx] for r_idx in r_idxs]  # [captions[r_idx*5] for r_idx in r_idxs]

        return d

    def plot_retrieval(d, tag, file_tag, q_lab, r_labs, qI, rI):

        def set_spines_color_width(ax, color, width):
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.spines['bottom'].set_color(color)
            ax.spines['top'].set_color(color)
            ax.spines['right'].set_color(color)
            ax.spines['left'].set_color(color)
            ax.spines['bottom'].set_linewidth(width)
            ax.spines['top'].set_linewidth(width)
            ax.spines['right'].set_linewidth(width)
            ax.spines['left'].set_linewidth(width)

        def get_q_rs():
            if tag == 'I2T':
                return Image.open(d['q']), d['r'], d['o']
            elif tag == 'T2I':
                return d['q'], [Image.open(i) for i in d['r']], Image.open(d['o'])
            elif tag == 'I2I':
                return Image.open(d['q']), [Image.open(i) for i in d['r']], d['o']
            elif tag == 'T2T':
                return d['q'], d['r'], Image.open(d['o'])

        colors = ['green' if q_lab == r_lab else 'red' for r_lab in r_labs]

        def print_results():
            print()
            print('Query:')
            print(q_idx, d['q'], d['o'])
            print()
            print('Retrieval:')
            for i, r in zip(r_idxs, d['r']):
                print(i, r)

        q, rs, o = get_q_rs()

        print_results()

        # figure size
        if tag.endswith('I'):
            fig = plt.figure(figsize=((len(rs) + 1) * 2.5, 4))
            subplots = len(rs) + 1
        else:
            fig = plt.figure(figsize=(12, 4))
            subplots = 2

        # plot query
        ax = fig.add_subplot(1, subplots, 1)
        ax.set_title('Query (idx:' + str(qI) + ')')
        if tag.startswith('I'):
            set_spines_color_width(ax, 'black', 3)
            plt.imshow(q)
            ax.text(0, 250, o)
        else:
            #ax.axis([0, len(rs), 0, len(rs)])
            plt.axis('off')
            plt.imshow(o)
            ax.text(0, 250, '(idx:' + str(qI) + ') ' + q)

        # plot responses
        if tag.endswith('I'):
            for i, r in enumerate(rs):
                ax = fig.add_subplot(1, subplots, 2 + i)
                ax.set_title('Response (idx:' + str(rI[i]) + ') ' + str(i + 1))
                set_spines_color_width(ax, colors[i], 3)
                plt.imshow(r)
        else:
            ax = fig.add_subplot(1, subplots, 2)
            for i, r in enumerate(rs):
                ax.axis([0, len(rs), 0, len(rs)])
                plt.axis('off')
                ax.text(0, i, '(idx:' + str(rI[i]) + ') ' + r, color=colors[i])

        # plt.tight_layout()
        plt.savefig(os.path.join('plots', ''.join([tag, file_tag, '.png'])))

    ham_dist_sorted_k, q_idx, r_idxs, q_lab, r_labs = get_retrieved_info(qB, rB, qL, rL, qI, rI, k)
    img_paths, captions = load_img_txt()

    d = get_retrieval_dict(img_paths, captions, q_idx, r_idxs, tag)

    plot_retrieval(d, tag, file_tag, q_lab, r_labs, q_idx, r_idxs)

    return d


def hr_hists(qBX, qBY, rBX, rBY, k=50, model=''):
    def hr_hist_data(qB, rB, k):
        n = len(qB)
        dicts = []
        max_hr = 0
        for i in range(n):

            ham_dist = calc_hamming_dist(qB[i, :], rB).squeeze().detach().cpu()
            ham_dist_sorted, idxs = torch.sort(ham_dist)

            ham_dist_sorted_k = ham_dist_sorted[:k].cpu().numpy()
            values, counts = np.unique(ham_dist_sorted_k, return_counts=True)

            temp_d = {}

            for v, c in zip(values, counts):
                temp_d[int(v)] = c
                if v > max_hr:
                    max_hr = int(v)
            dicts.append(temp_d)

        hr_list = []
        for hr in range(max_hr + 1):
            hr_list_temp = []
            for d in dicts:
                if hr in d.keys():
                    hr_list_temp.append(d[hr])
                else:
                    hr_list_temp.append(0)
            hr_list.append(np.array(hr_list_temp))

        return hr_list

    def plot_hr_hist(ds, tag, ax):
        bars = range(len(ds[0]))
        offsets = np.zeros(len(ds[0]))
        for i, d in enumerate(ds):
            ax.bar(bars, d, bottom=offsets, width=1, label=str(i))
            offsets = offsets + d
        plt.title(tag.upper(), size=20, weight='medium')
        ax.legend(title='HR', bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_xlabel('samples')
        ax.set_ylabel('K')
        plt.grid(axis='y')

    i2t = hr_hist_data(qBX, rBY, k)
    t2i = hr_hist_data(qBY, rBX, k)
    i2i = hr_hist_data(qBX, rBX, k)
    t2t = hr_hist_data(qBY, rBY, k)

    data = [i2t, t2i, i2i, t2t]
    tags = ['i2t', 't2i', 'i2i', 't2t']

    fig = plt.figure(figsize=(16, 8))
    for i, (tag, d) in enumerate(zip(tags, data)):
        ax = fig.add_subplot(2, 2, i + 1)
        plot_hr_hist(d, ', '.join([tag, model, 'k: {}'.format(k)]), ax)
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'hr_hists_' + model + '.png'))


def select_idxs(seq_length, n_to_select, n_from_select, seed=42):
    """
    Select n_to_select indexes from each consequent n_from_select indexes from range with length seq_length, split
    selected indexes to separate arrays

    input, range of length seq_length:
    range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    sequences of length n_from_select:
    sequences = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]

    selected n_to_select elements from each sequence
    selected = [[0, 4], [7, 9], [13, 14], [16, 18]]

    output, n_to_select lists of length seq_length / n_from_select:
    output = [[0, 7, 13, 16], [4, 9, 14, 18]]

    :param seq_length: length of sequence, say 10
    :param n_to_select: number of elements to select
    :param n_from_select: number of consequent elements
    :return:
    """
    random.seed(seed)
    idxs = [[] for _ in range(n_to_select)]
    for i in range(seq_length // n_from_select):
        ints = random.sample(range(n_from_select), n_to_select)
        for j in range(n_to_select):
            idxs[j].append(i * n_from_select + ints[j])
    return idxs
