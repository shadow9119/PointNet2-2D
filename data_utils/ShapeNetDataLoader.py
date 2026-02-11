'''
Modified by Sitong Chen
Date: Oct 2024
'''

# *_*coding:utf-8 *_*
import os, math
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
import torch
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    pc_min = np.empty(3, dtype=np.float64)
    pc_max = np.empty(3, dtype=np.float64)
    for i in range(pc.shape[1]):
        pc_min[i] = min(pc[:, i])
        pc_max[i] = max(pc[:, i])
        pc[:, i] = 2 * ((pc[:, i] - pc_min[i]) / (pc_max[i] - pc_min[i])) - 1
    return pc, pc_min, pc_max

class PartNormalDataset(Dataset):
    def __init__(self, root='./data', npoints=None, split='train', conf_channel=False):  # 将 npoints 默认值改为 None
        self.npoints = npoints  # 允许 npoints 为 None
        self.root = root
        self.split = split
        self.conf_channel = conf_channel

        # 初始化 datapath 列表
        self.datapath = []

        # 按照 split 直接从 JSON 文件中读取文件路径
        if self.split == 'trainval':
            with open(os.path.join(self.root, 'train_val_test_split/train_file_list.json'), 'r') as f:
                train_files = json.load(f)
            with open(os.path.join(self.root, 'train_val_test_split/val_file_list.json'), 'r') as f:
                val_files = json.load(f)
            fns = train_files + val_files
        elif self.split == 'train':
            with open(os.path.join(self.root, 'train_val_test_split/train_file_list.json'), 'r') as f:
                fns = json.load(f)
        elif self.split == 'val':
            with open(os.path.join(self.root, 'train_val_test_split/val_file_list.json'), 'r') as f:
                fns = json.load(f)
        elif self.split == 'test':
            with open(os.path.join(self.root, 'train_val_test_split/test_file_list.json'), 'r') as f:
                fns = json.load(f)
        else:
            print(f'Unknown split: {self.split}. Exiting..')
            exit(-1)

        # 检查文件扩展名是否为 .txt，并将其路径加入 self.datapath
        for fn in fns:
            if os.path.splitext(os.path.basename(fn))[1] == '.txt':
                self.datapath.append(fn)

        # 初始化缓存
        self.cache = {}
        self.cache_size = 200000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = np.array([0]).astype(np.int32)
            data = np.loadtxt(fn, delimiter=',').astype(np.float64)

            if not self.conf_channel:
                point_set = data[:, [0, 1]]  # use x, y
            else:
                point_set = data[:, [0, 1, 3]]  # use x, y, signal_conf
                point_set[:, -1] = point_set[:, -1].astype(np.int32)

            seg = data[:, -1].astype(np.int32)

            # 缓存数据
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        point_set_normalized = point_set
        point_set_normalized[:, 0:2], pc_min, pc_max = pc_normalize(point_set[:, 0:2])
        
        point_set_normalized_mask = np.full(len(seg), True, dtype=bool)  # 初始化为全True，默认为没有填充

        if self.npoints is not None:
            if len(seg) > self.npoints:
                choice = np.random.choice(len(seg), self.npoints, replace=False)
                point_set_normalized = point_set_normalized[choice, :]
                seg = seg[choice]
                point_set_normalized_mask = point_set_normalized_mask[choice]  # 更新mask
            elif len(seg) < self.npoints:
                if not self.conf_channel:
                    pad_point = np.ones((self.npoints - len(seg), 2), dtype=np.float32)
                else:
                    pad_point = np.ones((self.npoints - len(seg), 2), dtype=np.float32)
                    pad_conf = np.ones((self.npoints - len(seg), 1), dtype=np.int32)
                    pad_point = np.concatenate((pad_point, pad_conf), axis=1)

                point_set_normalized = np.concatenate((point_set_normalized, pad_point), axis=0)
                pad_seg = np.zeros(self.npoints - len(seg), dtype=np.int32)
                seg = np.concatenate((seg, pad_seg), axis=0)

                pad_point_bool = np.full(self.npoints - len(seg), False, dtype=bool)
                point_set_normalized_bool = np.full(len(seg), True, dtype=bool)
                point_set_normalized_mask = np.concatenate((point_set_normalized_bool, pad_point_bool))

        if self.split == 'test':
            return point_set_normalized, cls, seg, point_set_normalized_mask, pc_min, pc_max, fn

        return point_set_normalized, cls, seg


    def __len__(self):
        return len(self.datapath)
