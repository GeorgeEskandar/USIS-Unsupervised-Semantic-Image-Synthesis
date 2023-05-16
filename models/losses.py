import torch
import torch.nn.functional as F
import torch.nn as nn
from models.vggloss import VGG19
import math
import numpy as np

class losses_computer():
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

    def loss(self, input, label, for_real):
        #--- balancing classes ---
        weight_map = get_class_balancing(self.opt, input, label)
        #--- n+1 loss ---
        target = get_n1_target(self.opt, input, label, for_real)
        loss = F.cross_entropy(input, target, reduction='none')
        if for_real:
            loss = torch.mean(loss * weight_map[:, 0, :, :])
        else:
            loss = torch.mean(loss)
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask*output_D_real+(1-mask)*output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)


def get_class_balancing(opt, input, label):
    if not opt.no_balancing_inloss:
        class_occurence = torch.sum(label, dim=(0, 2, 3))
        if opt.contain_dontcare_label:
            class_occurence[0] = 0
        num_of_classes = (class_occurence > 0).sum()
        coefficients = torch.reciprocal(class_occurence) * torch.numel(label) / (num_of_classes * label.shape[1])
        integers = torch.argmax(label, dim=1, keepdim=True)
        if opt.contain_dontcare_label:
            coefficients[0] = 0
        weight_map = coefficients[integers]
    else:
        weight_map = torch.ones_like(input[:, :, :, :])
    return weight_map


def get_n1_target(opt, input, label, target_is_real):
    targets = get_target_tensor(opt, input, target_is_real)
    num_of_classes = label.shape[1]
    integers = torch.argmax(label, dim=1)
    targets = targets[:, 0, :, :] * num_of_classes
    integers += targets.long()
    integers = torch.clamp(integers, min=num_of_classes-1) - num_of_classes + 1
    return integers


def get_target_tensor(opt, input, target_is_real):
    if opt.gpu_ids != "-1":
        if target_is_real:
            return torch.cuda.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        else:
            return torch.cuda.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)
    else:
        if target_is_real:
            return torch.FloatTensor(1).fill_(1.0).requires_grad_(False).expand_as(input)
        else:
            return torch.FloatTensor(1).fill_(0.0).requires_grad_(False).expand_as(input)


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
                #print(loss.shape)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# Perceptual loss that uses a pretrained BDCN network
class BDCNLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(BDCNLoss, self).__init__()
        self.bdcn = BDCN().cuda()
        self.bdcn.load_state_dict(torch.load('%s' % ('./pretrained_models/final-model/bdcn_pretrained_on_bsds500.pth')))
        self.bdcn.eval()



    def forward(self, x, y):
        x_edge = self.bdcn(torch.argmax(x, dim=1, keepdim=True).repeat(1,3,1,1).float())[-1] #self.bdcn(x.repeat(1,3,1,1).float())
        x_edge = x_edge > torch.tensor([0]).cuda()
        x_edge = x_edge.float()
        '''     import matplotlib.pyplot as plt
        test = x_edge.detach().cpu()
        print(torch.max(test))
        print(torch.min(test))
        plt.imshow(test[0,0,:,:])'''
        y_edge = torch.sigmoid(self.bdcn(y)[-1])
        '''test = y_edge.clone().detach().cpu()
        print(torch.max(test))
        print(torch.min(test))
        plt.figure()
        plt.imshow(test[0,0,:,:])
        plt.show()'''
        loss = torch.mean(-(x_edge*y_edge))
        return loss

def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class BDCN(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=4):
        super(BDCN, self).__init__()
        self.pretrain = pretrain
        t = 1

        self.features = VGG16_C(pretrain, logger)
        self.msblock1_1 = MSBlock(64, rate)
        self.msblock1_2 = MSBlock(64, rate)
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock2_1 = MSBlock(128, rate)
        self.msblock2_2 = MSBlock(128, rate)
        self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = MSBlock(256, rate)
        self.msblock3_2 = MSBlock(256, rate)
        self.msblock3_3 = MSBlock(256, rate)
        self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock4_1 = MSBlock(512, rate)
        self.msblock4_2 = MSBlock(512, rate)
        self.msblock4_3 = MSBlock(512, rate)
        self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock5_1 = MSBlock(512, rate)
        self.msblock5_2 = MSBlock(512, rate)
        self.msblock5_3 = MSBlock(512, rate)
        self.conv5_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)

        self._initialize_weights(logger)

    def forward(self, x):
        features = self.features(x)
        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
                self.conv1_2_down(self.msblock1_2(features[1]))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)
        # print(s1.data.shape, s11.data.shape)
        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
            self.conv2_2_down(self.msblock2_2(features[3]))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        # print(s2.data.shape, s21.data.shape)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)
        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
            self.conv3_2_down(self.msblock3_2(features[5])) + \
            self.conv3_3_down(self.msblock3_3(features[6]))
        s3 = self.score_dsn3(sum3)
        s3 =self.upsample_4(s3)
        # print(s3.data.shape)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum3)
        s31 =self.upsample_4(s31)
        # print(s31.data.shape)
        s31 = crop(s31, x, 2, 2)
        sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
            self.conv4_2_down(self.msblock4_2(features[8])) + \
            self.conv4_3_down(self.msblock4_3(features[9]))
        s4 = self.score_dsn4(sum4)
        s4 = self.upsample_8(s4)
        # print(s4.data.shape)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum4)
        s41 = self.upsample_8(s41)
        # print(s41.data.shape)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
            self.conv5_2_down(self.msblock5_2(features[11])) + \
            self.conv5_3_down(self.msblock5_3(features[12]))
        s5 = self.score_dsn5(sum5)
        s5 = self.upsample_8_5(s5)
        # print(s5.data.shape)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum5)
        s51 = self.upsample_8_5(s51)
        # print(s51.data.shape)
        s51 = crop(s51, x, 0, 0)
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51

        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))

        return [p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2, fuse]

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)
        # print self.conv1_1_down.weight


class VGG16_C(nn.Module):
    """"""
    def __init__(self, pretrain=None, logger=None):
        super(VGG16_C, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)
        if pretrain:
            if '.npy' in pretrain:
                state_dict = np.load(pretrain).item()
                for k in state_dict:
                    state_dict[k] = torch.from_numpy(state_dict[k])
            else:
                state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    if logger:
                        logger.info('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 0.01)
        else:
            self._initialize_weights(logger)

    def forward(self, x):
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.relu2_1(self.conv2_1(pool1))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.relu3_1(self.conv3_1(pool2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        pool3 = self.pool3(conv3_3)
        conv4_1 = self.relu4_1(self.conv4_1(pool3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_3)
        # pool4 = conv4_3
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))

        side = [conv1_1, conv1_2, conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3, conv4_1,
                conv4_2, conv4_3, conv5_1, conv5_2, conv5_3]
        return side

    def _initialize_weights(self, logger=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
