"""
Author: zheng zhang
Email: musevr.ae@gmail.com
"""
import torch
from torch import nn

class WTFTP_AttnRemoved(nn.Module):
    def __init__(self, n_inp, n_oup, n_embding=128, n_encoderLayers=4, n_decoderLayers=4, proj='linear', activation='relu', maxlevel=1, en_dropout=0,
                 de_dropout=0, steps=None):
        super(WTFTP_AttnRemoved, self).__init__()
        self.args = locals()
        self.n_oup = n_oup
        self.maxlevel = maxlevel
        self.n_decoderLayers = n_decoderLayers
        self.steps = steps if steps is not None else [5,3,2]
        if activation == 'relu':
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding//2, bias=False, proj=proj), nn.ReLU(), Embed(n_embding//2, n_embding, bias=False, proj=proj), nn.ReLU())
        elif activation == 'sigmoid':
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding//2, bias=False, proj=proj), nn.Sigmoid(), Embed(n_embding//2, n_embding, bias=False, proj=proj), nn.Sigmoid())
        else:
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding//2, bias=False, proj=proj), nn.ReLU(), Embed(n_embding//2, n_embding, bias=False, proj=proj), nn.ReLU())

        self.encoder = nn.LSTM(input_size=n_embding,
                               hidden_size=n_embding,
                               num_layers=n_encoderLayers,
                               bidirectional=False,
                               batch_first=True,
                               dropout=en_dropout,
                               bias=False)
        self.decoders = nn.ModuleList([nn.LSTM(input_size=n_embding,
                                       hidden_size=n_embding,
                                       num_layers=n_decoderLayers,
                                       bidirectional=False,
                                       batch_first=True,
                                       dropout=de_dropout,
                                       bias=False) for _ in range(1+maxlevel)])
        self.LNs = nn.ModuleList([nn.LayerNorm(n_embding) for _ in range(1+maxlevel)])
        self.oup_embdings = nn.ModuleList([Embed(n_embding, n_oup, bias=True, proj=proj) for _ in range(1+maxlevel)])

    def forward(self, inp):
        """
        :param inp: shape: batch * n_sequence * n_attr
        :return: shape: batch * n_desiredLength * level * n_attr
        """
        inp = self.inp_embding(inp)
        _, (h_0, c_0) = self.encoder(inp)
        oup = []
        n_batch = inp.shape[0]
        i = 0
        for decoder, LN, oup_embding in zip(self.decoders, self.LNs, self.oup_embdings):
            this_decoder_oup = []
            decoder_first_inp = torch.zeros((n_batch, 1, inp.shape[-1]), device=inp.device)
            de_H = torch.zeros(self.n_decoderLayers, inp.shape[0], inp.shape[-1], dtype=torch.float,
                               device=inp.device)
            de_C = torch.zeros(self.n_decoderLayers, inp.shape[0], inp.shape[-1], dtype=torch.float,
                               device=inp.device)
            de_C[0, :, :] = c_0[-1, :, :].clone()
            decoder_hidden = (de_H, de_C)
            if i == self.maxlevel:
                this_step = self.steps[self.maxlevel-1]
            else:
                this_step = self.steps[i]
            for _ in range(this_step):
                decoder_first_inp, decoder_hidden = decoder(decoder_first_inp, decoder_hidden)
                de_oup = LN(decoder_first_inp)
                de_oup = oup_embding(de_oup)
                this_decoder_oup.append(de_oup)
            this_decoder_oup = torch.cat(this_decoder_oup, dim=1)  # shape: batch * n_desiredLength * n_attr
            oup.append(this_decoder_oup)
            i += 1
        return oup


class WTFTP(nn.Module):
    def __init__(self, n_inp, n_oup, his_step, n_embding=64, en_layers=4, de_layers=1, proj='linear',
                 activation='relu', maxlevel=1, en_dropout=0.,
                 de_dropout=0., out_split=False, decoder_init_zero=False, bias=False, se_skip=True,
                 attn_conv_params=None):
        """
        :param n_inp:
        :param n_oup:
        :param n_embding:
        :param layers:
        :param maxlevel:
        :param en_dropout:
        :param de_dropout:
        :param out_split:
        :param bias:
        """
        super(WTFTP, self).__init__()
        self.args = locals()
        self.n_embding = n_embding
        self.n_oup = n_oup
        self.maxlevel = maxlevel
        self.decoder_init_zero = decoder_init_zero
        self.de_layers = de_layers
        attn_conv_params = {0: {'stride': 2, 'kernel': 2, 'pad': 1}, 1: {'stride': 3, 'kernel': 3, 'pad': 0},
                           2: {'stride': 5, 'kernel': 5, 'pad': 1}} if attn_conv_params == None else attn_conv_params
        if activation == 'relu':
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding // 2, bias=False, proj=proj), nn.ReLU(),
                                             Embed(n_embding // 2, n_embding, bias=False, proj=proj), nn.ReLU())
        elif activation == 'sigmoid':
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding // 2, bias=False, proj=proj), nn.Sigmoid(),
                                             Embed(n_embding // 2, n_embding, bias=False, proj=proj), nn.Sigmoid())
        else:
            self.inp_embding = nn.Sequential(Embed(n_inp, n_embding // 2, bias=False, proj=proj), nn.ReLU(),
                                             Embed(n_embding // 2, n_embding, bias=False, proj=proj), nn.ReLU())
        self.encoder = nn.LSTM(input_size=n_embding,
                               hidden_size=n_embding,
                               num_layers=en_layers,
                               bidirectional=False,
                               batch_first=True,
                               dropout=en_dropout,
                               bias=False)
        self.Wt_attns = nn.ModuleList([WaveletAttention(hid_dim=n_embding, init_steps=his_step, skip=se_skip,
                                                        kernel=attn_conv_params[i]['kernel'],
                                                        stride=attn_conv_params[i]['stride'],
                                                        padding=attn_conv_params[i]['pad'])
                                       for i in range(maxlevel)] +
                                      [WaveletAttention(hid_dim=n_embding, init_steps=his_step, skip=se_skip,
                                                        kernel=attn_conv_params[maxlevel - 1]['kernel'],
                                                        stride=attn_conv_params[maxlevel - 1]['stride'],
                                                        padding=attn_conv_params[maxlevel - 1]['pad'])])
        self.decoders = nn.ModuleList([nn.LSTM(input_size=n_embding,
                                               hidden_size=n_embding,
                                               num_layers=de_layers,
                                               bidirectional=False,
                                               batch_first=True,
                                               dropout=de_dropout,
                                               bias=False) for _ in range(1 + maxlevel)])
        self.LNs = nn.ModuleList([nn.LayerNorm(n_embding) for _ in range(1 + maxlevel)])
        if out_split:
            assert n_embding % n_oup == 0
            self.out_split = out_split
            self.oup_embdings = nn.ModuleList(
                [nn.Linear(n_embding // n_oup, 1, bias=bias) for _ in range(n_oup * 2 ** maxlevel)])
        else:
            self.oup_embdings = nn.ModuleList(
                [Embed(n_embding, n_oup, bias=True, proj=proj) for _ in range(1 + maxlevel)])

    def forward(self, inp):
        """
        :param inp: shape: batch * n_sequence * n_attr
        :param steps: coeff length of wavelet
        :return: coef_set: shape: batch * steps * levels * n_oup,
                 all_scores_set: shape: batch * steps * levels * n_sequence, n_sequence here is timeSteps of inp
        """
        embdings = self.inp_embding(inp)  # batch * n_sequence * n_embding
        en_oup, (h_en, c_en) = self.encoder(embdings)
        all_de_oup_set = []
        all_scores_set = []
        for i, (attn, decoder, LN) in enumerate(zip(self.Wt_attns, self.decoders, self.LNs)):
            if self.decoder_init_zero:
                de_HC = None
            else:
                de_H = torch.zeros(self.de_layers, embdings.shape[0], embdings.shape[-1], dtype=torch.float,
                                   device=embdings.device)
                de_C = torch.zeros(self.de_layers, embdings.shape[0], embdings.shape[-1], dtype=torch.float,
                                   device=embdings.device)
                de_C[0, :, :] = c_en[-1, :, :].clone()
                de_HC = (de_H, de_C)
            de_inp, weight = attn(en_oup)
            de_oup_set, _ = decoder(de_inp, de_HC)  # shape: batch * steps * n_embding
            all_de_oup_set.append(LN(de_oup_set))
            all_scores_set.append(weight)
        all_scores_set = torch.cat(all_scores_set, dim=1)  # shape: batch * steps * levels * n_sequence
        if hasattr(self, 'out_split'):
            all_de_oup_set = torch.cat(all_de_oup_set, dim=2)  # shape: batch * steps * (levels*n_embding)
            split_all_de_oup_set = torch.split(all_de_oup_set, split_size_or_sections=self.n_embding // self.n_oup,
                                               dim=-1)
            coef_set = []
            for i, linear in enumerate(self.oup_embdings):
                coef_set.append(
                    linear(split_all_de_oup_set[i])
                )  # shape: batch * steps * 1
            coef_set = torch.cat(coef_set, dim=-1).reshape(inp.shape[0], -1, 2 ** self.maxlevel,
                                                           self.n_oup)  # shape: batch * steps * levels * n_oup
        else:
            coef_set = []
            for i, linear in enumerate(self.oup_embdings):
                coef_set.append(
                    linear(all_de_oup_set[i])
                )  # shape: batch * steps * n_oup
        return coef_set, all_scores_set

class Embed(nn.Module):
    def __init__(self, in_features, out_features, bias=True, proj='linear'):
        super(Embed, self).__init__()
        self.proj = proj
        if proj == 'linear':
            self.embed = nn.Linear(in_features, out_features, bias)
        else:
            self.embed = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1,
                                   padding=1,
                                   padding_mode='replicate', bias=bias)

    def forward(self, inp):
        """
        inp: B * T * D
        """
        if self.proj == 'linear':
            inp = self.embed(inp)
        else:
            inp = self.embed(inp.transpose(1, 2)).transpose(1, 2)
        return inp

class WaveletAttention(nn.Module):
    def __init__(self, hid_dim, init_steps, skip=True, kernel=3, stride=3, padding=0):
        super(WaveletAttention, self).__init__()
        self.enhance = EnhancedBlock(hid_size=hid_dim, channel=init_steps, skip=skip)
        self.convs = nn.Sequential(nn.Conv1d(in_channels=hid_dim,
                                             out_channels=hid_dim,
                                             kernel_size=kernel,
                                             stride=stride, padding=padding,
                                             padding_mode='zeros',
                                             bias=False),
                                   nn.ReLU())

    def forward(self, hid_set: torch.Tensor):
        """
        q = W1 * hid_0, K = W2 * hid_set
        softmax(q * K.T) * hid_set
        :param hid_set: Key set, shape: batch * n_sequence * hid_dim
        :param hid_0: Query hidden state, shape: batch * 1 * hid_dim
        :return: out: shape: shape: batch * 1 * hid_dim, scores: shape: batch * 1 * sequence
        """
        reweighted, weight = self.enhance(hid_set)
        align = self.convs(reweighted.transpose(1, 2)).transpose(1, 2)  # B * DestStep * D
        return align, weight


class EnhancedBlock(nn.Module):
    def __init__(self, hid_size, channel, skip=True):
        super(EnhancedBlock, self).__init__()
        self.skip = skip
        self.comp = nn.Sequential(nn.Linear(hid_size, hid_size // 2, bias=False),
                                  nn.ReLU(),
                                  nn.Linear(hid_size // 2, 1, bias=False))
        self.activate = nn.Sequential(nn.Linear(channel, channel // 2, bias=False),
                                      nn.ReLU(),
                                      nn.Linear(channel // 2, channel, bias=False),
                                      nn.Sigmoid())

    def forward(self, inp):
        S = self.comp(inp)  # B * T * 1
        E = self.activate(S.transpose(1, 2))  # B * 1 * T
        out = inp * E.transpose(1, 2).expand_as(inp)
        if self.skip:
            out += inp
        return out, E  # B * T * D

if __name__ == '__main__':
    model = WTFTP(6, 6, 9)
    model = WTFTP_AttnRemoved(6, 6)
    inp = torch.rand((100, 9, 6))
    out, _ = model(inp)
    print(model.__class__.__name__)
    print(out)
    print(model.args)
