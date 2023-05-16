import torch.nn as nn
import models.norms as norms
import torch
import torch.nn.functional as F
from models.discriminator import make_kernel,upfirdn2d,InverseHaarTransform,HaarTransform,ModulatedConv2d


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
        #self.conv_img =
        # (self.channels[-1])
        self.up = nn.Upsample(scale_factor=2)
        self.body = nn.ModuleList([])
        if self.opt.progressive_growing :
            self.torgbs = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
            if self.opt.progressive_growing:
                self.torgbs.append(ToRGB(in_channel=self.channels[i+1]))
        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, z=None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
            x = F.tanh(x)
        else :
            x = self.conv_img(F.leaky_relu(x, 2e-1))
            x = F.tanh(x)

        return x

    def forward_determinstic(self, input, noise_vector):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = noise_vector.to(dev)
            seg = torch.cat((z, seg), dim = 1)
        x = F.interpolate(seg, size=(self.init_W, self.init_H))
        x = self.fc(x)
        out = None
        for i in range(self.opt.num_res_blocks):
            x = self.body[i](x, seg)

            if self.opt.progressive_growing and out == None :
                out = self.torgbs[i](x)
            elif self.opt.progressive_growing :
                out = self.torgbs[i](x,skip = out)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)

        if self.opt.progressive_growing :
            x = out
            x = F.tanh(x)
        else :
            x = self.conv_img(F.leaky_relu(x, 2e-1))
            x = F.tanh(x)

        return x

class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):


        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))

        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel,opt, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        sp_norm = norms.get_spectral_norm(opt)

        if upsample:
            self.upsample = nn.Upsample(scale_factor = 2)

        self.conv = sp_norm(nn.Conv2d(in_channel, 3, 1, 1,padding_mode='reflect'))

    def forward(self, input, skip=None):
        out = self.conv(F.leaky_relu(input,2e1))

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class ToRGB_wavelet(nn.Module):
    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.iwt = InverseHaarTransform(3)
            self.upsample = Upsample(blur_kernel)
            self.dwt = HaarTransform(3)

        self.conv = nn.Conv2d(in_channel, 3 * 4,1,1)

    def forward(self, input, skip=None):
        out = self.conv(input)

        if skip is not None:
            skip = self.iwt(skip)
            skip = self.upsample(skip)
            skip = self.dwt(skip)

            out = out + skip

        return out

class IWT_Upsample_HWT(nn.Module):
    def __init__(self, factor=2,mode='nearest'):
        super().__init__()
        if factor != 2 :
            print('wavelet upsampling is not implemented for factors different than 2')

        self.factor = factor

        self.iwt = InverseHaarTransform(3)
        self.up = nn.Upsample(scale_factor=factor,mode=mode)
        self.hwt = HaarTransform(3)



    def forward(self, input):

        output = self.iwt(input)
        output = self.up(output)
        output = self.hwt(output)

        return output

class ConstantInput(nn.Module):
    def __init__(self, channel, size=(8,4)):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, *size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class ResidualWaveletGenerator_1(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        ch = opt.channels_G
        self.channels = [4*16*ch, 4*16*ch, 4*16*ch, 4*8*ch, 4*4*ch, 4*2*ch, 4*1*ch]

        self.init_W, self.init_H = self.compute_latent_vector_size(opt)

        self.conv_img = ToRGB_wavelet(in_channel=self.channels[-1],upsample = False)
        self.iwt = InverseHaarTransform(3)

        self.up = nn.Upsample(scale_factor=2)
        self.up_residual = IWT_Upsample_HWT(factor=2,mode='bilinear')
        self.body = nn.ModuleList([])
        for i in range(len(self.channels)-1):
            self.body.append(WaveletBlock_with_IWT_SPADE_HWT(self.channels[i], self.channels[i+1], opt))

        if not self.opt.no_3dnoise:
            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 4*16 * ch, 3, padding=1)
        else:
            self.fc = nn.Conv2d(self.opt.semantic_nc, 4*16 * ch, 3, padding=1)

#        self.constant_input = ConstantInput(self.channels[0],(self.init_W, self.init_H))

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h//2, w//2

    def forward(self, input, z=None):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
            seg = torch.cat((z, seg), dim = 1)


        x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x_s,x = self.body[i](x, seg)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
                x_s = self.up_residual(x_s)
                x = x+x_s


        x = self.conv_img(x+x_s)
        x = self.iwt(x)
        x = F.tanh(x)

        return x

    def forward_determinstic(self, input, noise_vector):
        seg = input
        if self.opt.gpu_ids != "-1":
            seg.cuda()
        if not self.opt.no_3dnoise:
            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
            z = noise_vector.to(dev)
            seg = torch.cat((z, seg), dim = 1)


        x = F.interpolate(seg, size=(self.init_W, self.init_H))

        x = self.fc(x)
        for i in range(self.opt.num_res_blocks):
            x_s,x = self.body[i](x, seg)

            if i < self.opt.num_res_blocks-1:
                x = self.up(x)
                x_s = self.up_residual(x_s)
                x = x+x_s


        x = self.conv_img(x+x_s)
        x = self.iwt(x)
        x = F.tanh(x)

        return x

class WaveletBlock_with_IWT_SPADE_HWT(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.IWT_SPADE_HWT(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.IWT_SPADE_HWT(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.IWT_SPADE_HWT(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):


        if self.learned_shortcut:
            x_s = self.conv_s(x)
            #x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x

        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        return x_s,dx





