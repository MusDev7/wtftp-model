import numpy as np

from pytorch_wavelets import DWT1DForward, DWT1DInverse
import torch
import pywt
import time
from torch.utils.data import DataLoader
import os


def wt_packet(inp, wavelet='db2', mode='symmetric', maxlevel=1):
    """
    the last-dim vector to be decomposed
    :param inp: shape: batch * n_attr * n_sequence
    :param wavelet:
    :param mode:
    :param maxlevel:
    :return: oup: shape : batch * n_attr * level * n_sequence
    """
    if maxlevel == 0:
        return inp.unsqueeze(2)
    dwt = DWT1DForward(wave=wavelet, J=1, mode=mode).to(inp.device)
    oup = [inp]
    for _ in range(maxlevel):
        tmp = []
        for item in oup:
            lo, hi = dwt(item)
            tmp += [lo, hi[0]]
        oup = tmp
    oup = torch.stack(oup, dim=2)
    return oup


def wt_packet_inverse(inp, wavelet='db2', mode='symmetric', maxlevel=1):
    """
    the last-dim vector to be composed
    :param inp: shape : batch * n_attr * level * n_sequence
    :param wavelet:
    :param mode:
    :param maxlevel:
    :return: oup: batch * n_attr * n_sequence
    """
    assert inp.shape[2] == 2**maxlevel
    if maxlevel == 0:
        return inp.squeeze(2)
    idwt = DWT1DInverse(wave=wavelet, mode=mode).to(inp.device)
    oup = inp
    for level in range(maxlevel, 0, -1):
        tmp = []
        for i in range(2**(level-1)):
            lo, hi = oup[:, :, 2 * i, :], oup[:, :, 2 * i + 1, :]
            hi = [hi]
            tmp.append(idwt((lo, hi)))
        oup = torch.stack(tmp, dim=2)
    oup = oup.squeeze(2)
    return oup


def progress_bar(step, n_step, str, start_time=time.perf_counter(), bar_len=20):
    '''
    :param bar_len: length of the bar
    :param step: from 0 to n_step-1
    :param n_step: number of steps
    :param str: info to be printed
    :param start_time: time to begin the progress_bar
    :return:
    '''
    step = step+1
    a = "*" * int(step * bar_len / n_step)
    b = " " * (bar_len - int(step * bar_len / n_step))
    c = step / n_step * 100
    dur = time.perf_counter() - start_time
    print("\r{:^3.0f}%[{}{}]{:.2f}s {}".format(c, a, b, dur, str), end="")
    if step == n_step:
        print('')


def wt_coef_len(in_length, wavelet, mode, maxlevel):
    test_inp = torch.ones((1, 1, in_length))
    test_oup = wt_packet(test_inp, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    return test_oup.shape[-1]


class recorder:
    def __init__(self, *attrs):
        if attrs is not None:
            self.saver = {}
            self.attrs = attrs
            for attr in attrs:
                self.saver[attr] = []
        else:
            self.saver = {}
            self.attrs = attrs

    def add(self, attr, val):
        self.saver[attr].append(val)

    def __getitem__(self, item):
        return self.saver[item]

    def __len__(self):
        return len(self.attrs)

    def push(self, path, filename='recorde.pt'):
        torch.save(self.saver, os.path.join(path, filename))

    def pull(self, filepath):
        self.saver = torch.load(filepath)
        self.attrs = self.saver.keys()
