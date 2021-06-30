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


def generate_hashes_from_dataloader(train_loader, test_loader, model_I, model_T, label_dim):
    # response hashes / labels
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, data_T, lab, _) in enumerate(train_loader):
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
    for _, (data_I, data_T, lab, _) in enumerate(test_loader):
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

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


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
        # total (TP + FP): total[k] is count of distances less or equal to k (from current query sample to retrieval samples)
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


def top_k_hists(qBX, qBY, rBX, rBY, k=20, model=''):
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
        ax.bar(x, y, width=1)
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
