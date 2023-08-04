import argparse
import os
import random
import time
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from dataloader import DataGenerator
import logging
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch_wavelets import DWT1DForward, DWT1DInverse

parser = argparse.ArgumentParser()
parser.add_argument('--minibatch_len', default=10, type=int)
parser.add_argument('--pre_len', default=1, type=int)
parser.add_argument('--interval', default=1, type=int)
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--logdir', default='./log', type=str)
parser.add_argument('--datadir', default='dataaaaaa', type=str)
parser.add_argument('--netdir', default=None, type=str)


class Test:
    def __init__(self, opt, net=None):
        self.opt = opt
        self.iscuda = torch.cuda.is_available()
        self.device = f'cuda:{torch.cuda.current_device()}' if self.iscuda and not opt.cpu else 'cpu'
        self.data_set = DataGenerator(data_path=self.opt.datadir,
                                      minibatch_len=opt.minibatch_len, interval=opt.interval,
                                      use_preset_data_ranges=False, train=False, dev=False, test_shuffle=True)
        self.net = net
        self.model_path = None
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.MAE = torch.nn.L1Loss(reduction='mean')
        if net is not None:
            assert next(self.net.parameters()).device == self.device

    def load_model(self, model_path):
        self.model_path = model_path
        self.net = torch.load(model_path, map_location=self.device)

    def test(self):
        print_str = f'model details:\n{self.net.args}'
        print(print_str)
        self.log_path = self.opt.logdir + f'/{datetime.datetime.now().strftime("%y-%m-%d")}'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        logging.basicConfig(filename=os.path.join(self.log_path, f'test_{self.net.args["train_opt"].comments}.log'),
                            filemode='w', format='%(asctime)s   %(message)s', level=logging.DEBUG)
        logging.debug(print_str)
        logging.debug(self.model_path)
        test_data = DataLoader(dataset=self.data_set.test_set, batch_size=self.opt.batch_size, shuffle=False,
                               collate_fn=self.data_set.collate)
        idwt = DWT1DInverse(wave=self.net.args['train_opt'].wavelet, mode=self.net.args['train_opt'].wt_mode).to(
            self.device)
        self.net.eval()
        tgt_set = []
        pre_set = []

        with torch.no_grad():
            his_batch_set = []
            all_score_set = []
            for i, batch in enumerate(test_data):
                batch = torch.FloatTensor(batch).to(self.device)
                n_batch, _, n_attr = batch.shape
                inp_batch = batch[:, :self.opt.minibatch_len-self.opt.pre_len, :]  # shape: batch * his_len * n_attr
                his_batch_set.append(inp_batch)
                tgt_set.append(batch[:, -self.opt.pre_len:, :])  # shape: batch * pre_len * n_attr
                pre_batch_set = []
                for j in range(self.opt.pre_len):
                    if j > 0:
                        new_batch = pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :].unsqueeze(1)
                        inp_batch = torch.cat((inp_batch[:, 1:, :], new_batch), dim=1)  # shape: batch * his_len * n_attr
                    if self.net.__class__.__name__ == 'WTFTP':
                        wt_pre_batch, score_set = self.net(inp_batch)
                    else:
                        wt_pre_batch = self.net(inp_batch)
                    pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                      [comp.transpose(1, 2).contiguous() for comp in
                                       wt_pre_batch[:-1]])).contiguous()
                    pre_batch = pre_batch.transpose(1, 2)  # shape: batch * n_sequence * n_attr
                    pre_batch_set.append(pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :])
                if self.net.__class__.__name__ == 'WTFTP' and j == 0:
                    all_score_set.append(score_set)
                pre_batch_set = torch.stack(pre_batch_set, dim=1)  # shape: batch * pre_len * n_attr
                pre_set.append(pre_batch_set)

            tgt_set = torch.cat(tgt_set, dim=0)
            pre_set = torch.cat(pre_set, dim=0)
            # try:
            #     all_score_set = torch.cat(all_score_set, dim=0)
            # except:
            #     all_score_set = ['no scores']
            # his_batch_set = torch.cat(his_batch_set, dim=0)
            # torch.save([his_batch_set, tgt_set, pre_set, all_score_set],
            #            f'{self.net.args["train_opt"].comments}_his_tgt_pre_score.pt', _use_new_zipfile_serialization=False)
            for i in range(self.opt.pre_len):
                avemse = float(self.MSE(tgt_set[:, :i + 1, :], pre_set[:, :i + 1, :]).cpu())
                avemae = float(self.MAE(tgt_set[:, :i + 1, :], pre_set[:, :i + 1, :]).cpu())
                rmse = {}
                mae = {}
                mre = {}
                for j, name in enumerate(self.data_set.attr_names):
                    rmse[name] = float(self.MSE(self.data_set.unscale(tgt_set[:, :i + 1, j], name),
                                                self.data_set.unscale(pre_set[:, :i + 1, j], name)).sqrt().cpu())
                    mae[name] = float(self.MAE(self.data_set.unscale(tgt_set[:, :i + 1, j], name),
                                               self.data_set.unscale(pre_set[:, :i + 1, j], name)).cpu())
                    logit = self.data_set.unscale(tgt_set[:, :i + 1, j], name) != 0
                    mre[name] = float(torch.mean(torch.abs(self.data_set.unscale(tgt_set[:, :i + 1, j], name)-
                                                self.data_set.unscale(pre_set[:, :i + 1, j], name))[logit]/
                                      self.data_set.unscale(tgt_set[:, :i + 1, j], name)[logit]).cpu()) * 100 if name in 'lonlatalt' \
                                else "N/A"
                lon = self.data_set.unscale(pre_set[:, :i + 1, 0], 'lon').cpu().numpy()
                lat = self.data_set.unscale(pre_set[:, :i + 1, 1], 'lat').cpu().numpy()
                alt = self.data_set.unscale(pre_set[:, :i + 1, 2], 'alt').cpu().numpy() / 100  # km
                X, Y, Z = self.gc2ecef(lon, lat, alt)
                lon_t = self.data_set.unscale(tgt_set[:, :i + 1, 0], 'lon').cpu().numpy()
                lat_t = self.data_set.unscale(tgt_set[:, :i + 1, 1], 'lat').cpu().numpy()
                alt_t = self.data_set.unscale(tgt_set[:, :i + 1, 2], 'alt').cpu().numpy() / 100  # km
                X_t, Y_t, Z_t = self.gc2ecef(lon_t, lat_t, alt_t)
                MDE = np.mean(np.sqrt((X - X_t) ** 2 + (Y - Y_t) ** 2 + (Z - Z_t) ** 2))
                print_str = f'\nStep {i + 1}: \naveMSE(scaled): {avemse:.8f}, in each attr(RMSE, unscaled): {rmse}\n' \
                            f'aveMAE(scaled): {avemae:.8f}, in each attr(MAE, unscaled): {mae}\n' \
                            f'In each attr(MRE, %): {mre}\n' \
                            f'MDE(unscaled): {MDE:.8f}\n'
                print(print_str)
                logging.debug(print_str)

    def gc2ecef(self, lon, lat, alt):
        a = 6378.137  # km
        b = 6356.752
        lat = np.radians(lat)
        lon = np.radians(lon)
        e_square = 1 - (b ** 2) / (a ** 2)
        N = a / np.sqrt(1 - e_square * (np.sin(lat) ** 2))
        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = ((b ** 2) / (a ** 2) * N + alt) * np.sin(lat)
        return X, Y, Z

    def draw_demo(self, items=None, realtime=False):
        plt.rcParams['font.family'] = 'arial'
        plt.rcParams['font.size'] = 16
        total_num = len(self.data_set.test_set)
        test_data = DataLoader(dataset=self.data_set.test_set, batch_size=self.opt.batch_size, shuffle=False,
                               collate_fn=self.data_set.collate)
        idwt = DWT1DInverse(wave=self.net.args['train_opt'].wavelet, mode=self.net.args['train_opt'].wt_mode).to(
            self.device)
        if items is None:
            items = [random.randint(0, total_num)]
        elif type(items) is int:
            items = [items % total_num]
        elif type(items) is list and len(items) > 0 and type(items[0]) is int:
            pass
        else:
            TypeError(type(items))
        if realtime:
            items = [int(input("item: "))]
        while len(items) > 0 and items[0] > 0:
            item = items[0]
            del items[0]
            n_batch = item // self.opt.batch_size
            n_minibatch = item % self.opt.batch_size
            sel_batch = None
            for i, batch in enumerate(test_data):
                if i == n_batch:
                    sel_batch = batch
                    break
            traj = sel_batch[n_minibatch:n_minibatch + 1, ...]
            with torch.no_grad():
                self.net.eval()
                self.net.to(self.device)
                inp_batch = torch.FloatTensor(traj[:, :self.opt.minibatch_len-self.opt.pre_len, :]).to(
                    self.device)  # shape: 1 * his_len * n_attr
                pre_batch_set = []
                full_pre_set = []
                for j in range(self.opt.pre_len):
                    if j > 0:
                        new_batch = pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :].unsqueeze(1)
                        inp_batch = torch.cat((inp_batch[:, 1:, :], new_batch), dim=1)  # shape: batch * his_len * n_attr
                    if self.net.__class__.__name__ == 'WTFTP':
                        wt_pre_batch, score_set = self.net(inp_batch)
                    else:
                        wt_pre_batch = self.net(inp_batch)
                    if j == 0:
                        first_wt_pre = wt_pre_batch
                    pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                      [comp.transpose(1, 2).contiguous() for comp in
                                       wt_pre_batch[:-1]])).contiguous()
                    pre_batch = pre_batch.transpose(1, 2)  # shape: 1 * n_sequence * n_attr
                    pre_batch_set.append(pre_batch[:, self.opt.minibatch_len-self.opt.pre_len, :])
                    full_pre_set.append(pre_batch[:, :self.opt.minibatch_len-self.opt.pre_len + 1, :].clone())
                pre_batch_set = torch.stack(pre_batch_set, dim=1)  # shape: 1 * pre_len * n_attr

            lla_his = np.array(traj[0, :self.opt.minibatch_len-self.opt.pre_len, 0:3])  # shape: his_len * n_attr
            lla_trg = np.array(traj[0, -self.opt.pre_len:, 0:3])  # shape: pre_len * n_attr
            lla_pre = np.array(pre_batch_set[0, :, 0:3].cpu().numpy())  # shape: pre_len * n_attr
            for i, name in enumerate(self.data_set.attr_names):
                if i > 2:
                    break
                lla_his[:, i] = self.data_set.unscale(lla_his[:, i], name)
                lla_trg[:, i] = self.data_set.unscale(lla_trg[:, i], name)
                lla_pre[:, i] = self.data_set.unscale(lla_pre[:, i], name)

            fig = plt.figure(figsize=(9,9))
            elev_azim_set = [[90, 0], [0, 0], [0, 90], [None, None]]  # represent top view, lateral view(lat), lateral view(lon) and default, respectively
            for i, elev_azim in enumerate(elev_azim_set):
                ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
                ax.plot3D(lla_his[:, 0], lla_his[:, 1], lla_his[:, 2], marker='o', markeredgecolor='dodgerblue',
                          label='his')
                ax.plot3D(lla_trg[:, 0], lla_trg[:, 1], lla_trg[:, 2], marker='*', markeredgecolor='blueviolet',
                          label='tgt')
                ax.plot3D(lla_pre[:, 0], lla_pre[:, 1], lla_pre[:, 2], marker='p', markeredgecolor='orangered',
                          label='pre')
                ax.set_xlabel('lon')
                ax.set_ylabel('lat')
                ax.set_zlabel('alt')
                ax.set_zlim(min(lla_his[:, 2]) - 20, max(lla_his[:, 2]) + 20)
            plt.suptitle(f'item_{item}')
            ax.legend()
            plt.tight_layout()
            plt.show()
            if realtime:
                items.append(int(input("item: ")))

if __name__ == '__main__':
    opt = parser.parse_args()
    test = Test(opt)
    test.load_model(opt.netdir)
    test.test()
