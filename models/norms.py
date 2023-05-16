import torch.nn.utils.spectral_norm as spectral_norm
from models.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn as nn
import torch.nn.functional as F
import torch

class SPADE(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class IWT_SPADE_HWT(nn.Module):
    def __init__(self, opt, norm_nc, label_nc):
        super().__init__()
        self.first_norm = get_norm_layer(opt, norm_nc//4)
        ks = opt.spade_ks
        nhidden = 128
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc//4, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc//4, kernel_size=ks, padding=pw)
        self.iwt = InverseHaarTransform(3)
        self.hwt = HaarTransform(3)

    def forward(self, x, segmap):
        x = self.iwt(x)
        normalized = self.first_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        out = self.hwt(out)
        return out

def get_spectral_norm(opt):
    if opt.no_spectral_norm:
        return torch.nn.Identity()
    else:
        return spectral_norm

def get_norm_layer(opt, norm_nc):
    if opt.param_free_norm == 'instance':
        return nn.InstanceNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'syncbatch':
        return SynchronizedBatchNorm2d(norm_nc, affine=False)
    if opt.param_free_norm == 'batch':
        return nn.BatchNorm2d(norm_nc, affine=False)
    else:
        raise ValueError('%s is not a recognized param-free norm type in SPADE'
                         % opt.param_free_norm)

class HaarTransform(nn.Module):
    def __init__(self, in_channels,four_channels = True,levels =1):
        super().__init__()


        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)
        self.four_channels = four_channels

        if levels > 1 and not four_channels :
            self.next_level = HaarTransform(in_channels,four_channels,levels-1)
        else :
            self.next_level = None

    def forward(self, input):

        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        if self.next_level != None :
            ll = self.next_level(ll)

        if self.four_channels :
            return torch.cat((ll, lh, hl, hh), 1)
        else :
            return torch.cat((torch.cat((ll,lh),-2),torch.cat((hl,hh),-2)),-1)

class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels,four_channels = True,levels = 1):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

        self.four_channels = four_channels

        if levels > 1 and not four_channels :
            self.next_level = InverseHaarTransform(in_channels,four_channels,levels-1)
        else :
            self.next_level = None

    def forward(self, input):
        if self.four_channels :
            ll, lh, hl, hh = input.chunk(4, 1)
        else :
            toprow,bottomrow = input.chunk(2,-1)
            ll,lh = toprow.chunk(2,-2)
            hl,hh = bottomrow.chunk(2,-2)

        if self.next_level != None :
            ll = self.next_level(ll)

        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh

def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(
        out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        :,
        max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),
    ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])