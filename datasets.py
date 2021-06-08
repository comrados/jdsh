import torch
from PIL import Image
from args import cfg
import numpy as np
import scipy.io as scio
from torchvision import transforms
import h5py
import random

if cfg.DATASET == "UCM":

    def load_ucm_train_query_db():
        random.seed(cfg.SEED)
        images, captions, labels = load_ucm()

        train, query, db = split_ucm(images, captions, labels)

        return train, query, db


    def load_ucm():
        with h5py.File(cfg.DATASET_PATH, "r") as hf:
            images = hf['image_emb'][:]
            images = (images - images.mean()) / images.std()

            captions = hf['caption_emb'][:]
            captions = (captions - captions.mean()) / captions.std()

            labels = hf['classcodes'][:]

        return images, captions, labels


    def split_ucm(images, captions, labels):
        idx_train, idx_query, idx_db = get_split_idxs(len(images))
        idx_train_cap, idx_query_cap, idx_db_cap = get_caption_idxs(idx_train, idx_query, idx_db)

        train = images[idx_train], captions[idx_train_cap], labels[idx_train]
        query = images[idx_query], captions[idx_query_cap], labels[idx_query]
        db = images[idx_db], captions[idx_db_cap], labels[idx_db]

        return train, query, db


    def get_split_idxs(arr_len):
        idx_all = list(range(arr_len))
        idx_train, idx_eval = split_indexes(idx_all, cfg.DATASET_TRAIN_SPLIT)
        idx_query, idx_db = split_indexes(idx_eval, cfg.DATASET_QUERY_SPLIT)

        return idx_train, idx_query, idx_db


    def split_indexes(idx_all, split):
        idx_length = len(idx_all)
        selection_length = int(idx_length * split)

        idx_selection = sorted(random.sample(idx_all, selection_length))

        idx_rest = sorted(list(set(idx_all).difference(set(idx_selection))))

        return idx_selection, idx_rest


    def get_caption_idxs(idx_train, idx_query, idx_db):
        idx_train_cap = get_caption_idxs_from_img_idxs(idx_train)
        idx_query_cap = get_caption_idxs_from_img_idxs(idx_query)
        idx_db_cap = get_caption_idxs_from_img_idxs(idx_db)
        return idx_train_cap, idx_query_cap, idx_db_cap


    def get_caption_idxs_from_img_idxs(img_idxs):
        caption_idxs = []
        for idx in img_idxs:
            for i in range(5):  # each image has 5 captions
                caption_idxs.append(idx * 5 + i)
        return caption_idxs


    train, query, db = load_ucm_train_query_db()

    txt_feat_len = train[1].shape[1]
    img_feat_len = train[0].shape[1]


    class UCM5(torch.utils.data.Dataset):

        def __init__(self, type='train'):

            if type == 'train':
                self.images, self.captions, self.labels = train
            elif type == 'db':
                self.images, self.captions, self.labels = db
            elif type == 'query':
                self.images, self.captions, self.labels = query
            else:
                raise Exception('wrong type')

        def __getitem__(self, index):
            idx_img, idx_txt = self.get_idx_combination_duplet(index)

            txt = self.captions[idx_txt]
            target = self.labels[idx_img]
            img = self.images[idx_img]

            return img, txt, target, index

        def __len__(self):
            return len(self.captions)

        @staticmethod
        def get_idx_combination_duplet(index):
            return index // 5, index


    class UCM2(torch.utils.data.Dataset):

        def __init__(self, type='train'):
            self.type = type

            if type == 'train':
                self.images, self.captions, self.labels = train
                self.captions = self.captions[self.randomly_select_caption_indexes()]
            elif type == 'db':
                self.images, self.captions, self.labels = db
            elif type == 'query':
                self.images, self.captions, self.labels = query
            else:
                raise Exception('wrong type')

        def __getitem__(self, index):
            idx_img, idx_txt = self.get_idx_combination_duplet(index)

            txt = self.captions[idx_txt]
            target = self.labels[idx_img]
            img = self.images[idx_img]

            return img, txt, target, index

        def __len__(self):
            return len(self.captions)

        def get_idx_combination_duplet(self, index):
            if self.type == 'train':
                return index // 2, index
            else:
                return index // 5, index

        def randomly_select_caption_indexes1(self):
            random.seed(cfg.SEED)
            idxs = []
            for i in range(len(self.images)):
                ints = random.sample(range(5), 2)
                idxs.append(i * 5 + ints[0])
                idxs.append(i * 5 + ints[1])
            return idxs

        def randomly_select_caption_indexes(self):
            random.seed(cfg.SEED)
            idxs = []
            for i in range(len(self.images)):
                ints = random.sample(range(5), 1)
                idxs.append(i * 5 + ints[0])
            return idxs

if cfg.DATASET == "UCM_":

    def load_ucm(path):
        with h5py.File(path, "r") as hf:
            images = hf['image_emb'][:]
            images = (images - images.mean()) / images.std()

            captions = hf['caption_emb'][:]
            captions = (captions - captions.mean()) / captions.std()

            labels = hf['classcodes'][:]

        return images, captions, labels


    img_set, txt_set, label_set = load_ucm(cfg.DATASET_PATH)

    indices = list(range(len(txt_set)))
    np.random.seed(cfg.SEED)
    np.random.shuffle(indices)

    indexTest = indices[:cfg.QUERY_SIZE]
    indexDatabase = indices[cfg.QUERY_SIZE:]
    indexTrain = indices[:cfg.TRAIN_SIZE]

    indexTest_img = [i // 5 for i in indexTest]
    indexDatabase_img = [i // 5 for i in indexDatabase]
    indexTrain_img = [i // 5 for i in indexTrain]

    txt_feat_len = txt_set.shape[1]
    img_feat_len = img_set.shape[1]


    class UCM(torch.utils.data.Dataset):
        def __init__(self, train=True, database=False):

            if train:
                self.element_indices = indexTrain
                self.train_labels = label_set[indexTrain_img]
            elif database:
                self.element_indices = indexDatabase
                self.train_labels = label_set[indexDatabase_img]
            else:
                self.element_indices = indexTest
                self.train_labels = label_set[indexTest_img]

        def __getitem__(self, index):
            real_data_index = self.element_indices[index]
            real_idx_img, real_idx_txt = self.get_idx_combination_duplet(real_data_index)

            txt = txt_set[real_idx_txt]
            target = label_set[real_idx_img]
            img = img_set[real_idx_img]

            return img, txt, target, index

        def __len__(self):
            return len(self.element_indices)

        @staticmethod
        def get_idx_combination_duplet(index):
            return index // 5, index

if cfg.DATASET == "MIRFlickr":

    label_set = scio.loadmat(cfg.LABEL_DIR)
    label_set = np.array(label_set['LAll'], dtype=np.float)
    txt_set = scio.loadmat(cfg.TXT_DIR)
    txt_set = np.array(txt_set['YAll'], dtype=np.float)

    first = True
    for label in range(label_set.shape[1]):
        index = np.where(label_set[:, label] == 1)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:160]
            train_index = index[160:160 + 400]
            first = False
        else:
            ind = np.array([i for i in list(index) if i not in (list(train_index) + list(test_index))])
            test_index = np.concatenate((test_index, ind[:80]))
            train_index = np.concatenate((train_index, ind[80:80 + 200]))

    database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])

    if train_index.shape[0] < 5000:
        pick = np.array([i for i in list(database_index) if i not in list(train_index)])
        N = pick.shape[0]
        perm = np.random.permutation(N)
        pick = pick[perm]
        res = 5000 - train_index.shape[0]
        train_index = np.concatenate((train_index, pick[:res]))

    indexTest = test_index
    indexDatabase = database_index
    indexTrain = train_index

    mir_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mir_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = txt_set.shape[1]


    class MIRFlickr(torch.utils.data.Dataset):
        def __init__(self, transform=None, target_transform=None, train=True, database=False):
            self.transform = transform
            self.target_transform = target_transform

            if train:
                self.train_labels = label_set[indexTrain]
                self.train_index = indexTrain
                self.txt = txt_set[indexTrain]
            elif database:
                self.train_labels = label_set[indexDatabase]
                self.train_index = indexDatabase
                self.txt = txt_set[indexDatabase]
            else:
                self.train_labels = label_set[indexTest]
                self.train_index = indexTest
                self.txt = txt_set[indexTest]

        def __getitem__(self, index):

            mirflickr = h5py.File(cfg.IMG_DIR, 'r', libver='latest', swmr=True)
            img, target = mirflickr['IAll'][self.train_index[index]], self.train_labels[index]
            img = Image.fromarray(np.transpose(img, (2, 1, 0)))
            mirflickr.close()

            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)

if cfg.DATASET == "NUSWIDE":

    label_set = scio.loadmat(cfg.LABEL_DIR)
    label_set = np.array(label_set['LAll'], dtype=np.float)
    txt_file = h5py.File(cfg.TXT_DIR, 'r')
    txt_set = np.array(txt_file['YAll']).transpose()
    txt_file.close()

    first = True

    for label in range(label_set.shape[1]):
        index = np.where(label_set[:, label] == 1)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:200]
            train_index = index[200:700]
            first = False
        else:
            ind = np.array([i for i in list(index) if i not in (list(train_index) + list(test_index))])
            test_index = np.concatenate((test_index, ind[:200]))
            train_index = np.concatenate((train_index, ind[200:700]))

    database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])

    indexTest = test_index
    indexDatabase = database_index
    indexTrain = train_index

    nus_train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    nus_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    txt_feat_len = txt_set.shape[1]


    class NUSWIDE(torch.utils.data.Dataset):

        def __init__(self, transform=None, target_transform=None, train=True, database=False):
            self.transform = transform
            self.target_transform = target_transform
            if train:
                self.train_labels = label_set[indexTrain]
                self.train_index = indexTrain
                self.txt = txt_set[indexTrain]
            elif database:
                self.train_labels = label_set[indexDatabase]
                self.train_index = indexDatabase
                self.txt = txt_set[indexDatabase]
            else:
                self.train_labels = label_set[indexTest]
                self.train_index = indexTest
                self.txt = txt_set[indexTest]

        def __getitem__(self, index):

            nuswide = h5py.File(cfg.IMG_DIR, 'r', libver='latest', swmr=True)
            img, target = nuswide['IAll'][self.train_index[index]], self.train_labels[index]
            img = Image.fromarray(np.transpose(img, (2, 1, 0)))
            nuswide.close()

            txt = self.txt[index]

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, txt, target, index

        def __len__(self):
            return len(self.train_labels)
