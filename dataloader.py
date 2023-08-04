import json
import logging
import os
import random
# import coordinate_conversion as cc
import numpy as np
import torch
import torch.utils.data as tu_data

class DataGenerator:
    def __init__(self, data_path, minibatch_len, interval=1, use_preset_data_ranges=False,
                 train=True, test=True, dev=True, train_shuffle=True, test_shuffle=False, dev_shuffle=True):
        assert os.path.exists(data_path)
        self.attr_names = ['lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz']
        self.data_path = data_path
        self.interval = interval
        self.minibatch_len = minibatch_len
        self.data_status = np.load('data_ranges.npy', allow_pickle=True).item()
        assert type(self.data_status) is dict
        self.preset_data_ranges = {"lon": {"max": 113.689, "min": 93.883}, "lat": {"max": 37.585, "min": 19.305},
                                           "alt": {"max": 1500, "min": 0}, "spdx": {"max": 878, "min": -945},
                                           "spdy": {"max": 925, "min": -963}, "spdz": {"max": 43, "min": -48}}
        self.use_preset_data_ranges = use_preset_data_ranges
        if train:
            self.train_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'train'), shuffle=train_shuffle))
        if dev:
            self.dev_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'dev'), shuffle=dev_shuffle))
        if test:
            self.test_set = mini_DataGenerator(self.readtxt(os.path.join(self.data_path, 'test'), shuffle=test_shuffle))
        if use_preset_data_ranges:
            assert self.preset_data_ranges is not None
        print('data range:', self.data_status)

    def readtxt(self, data_path, shuffle=True):
        assert os.path.exists(data_path)
        data = []
        for root, dirs, file_names in os.walk(data_path):
            for file_name in file_names:
                if not file_name.endswith('txt'):
                    continue
                with open(os.path.join(root, file_name)) as file:
                    lines = file.readlines()
                    lines = lines[::self.interval]
                    if len(lines) == self.minibatch_len:
                        data.append(lines)
                    elif len(lines) < self.minibatch_len:
                        continue
                    else:
                        for i in range(len(lines)-self.minibatch_len+1):
                            data.append(lines[i:i+self.minibatch_len])
        print(f'{len(data)} items loaded from \'{data_path}\'')
        if shuffle:
            random.shuffle(data)
        return data

    def scale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_status if not self.use_preset_data_ranges else self.preset_data_ranges
        inp = (inp-data_status[attr]['min'])/(data_status[attr]['max']-data_status[attr]['min'])
        return inp

    def unscale(self, inp, attr):
        assert type(attr) is str and attr in self.attr_names
        data_status = self.data_status if not self.use_preset_data_ranges else self.preset_data_ranges
        inp = inp*(data_status[attr]['max']-data_status[attr]['min'])+data_status[attr]['min']
        return inp

    def collate(self, inp):
        '''
        :param inp: batch * n_sequence * n_attr
        :return:
        '''
        oup = []
        for minibatch in inp:
            tmp = []
            for line in minibatch:
                items = line.strip().split("|")
                lon, lat, alt, spdx, spdy, spdz = float(items[4]), float(items[5]), int(float(items[6]) / 10), \
                                                  float(items[7]), float(items[8]), float(items[9])
                tmp.append([lon, lat, alt, spdx, spdy, spdz])
            minibatch = np.array(tmp)
            for i in range(minibatch.shape[-1]):
                minibatch[:, i] = self.scale(minibatch[:, i], self.attr_names[i])
            oup.append(minibatch)
        return np.array(oup)


class mini_DataGenerator(tu_data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
