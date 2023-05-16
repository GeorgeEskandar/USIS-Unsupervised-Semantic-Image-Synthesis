from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
from torch.nn import init
import models.losses as losses
from models.CannyFilter import CannyFilter
from torch import nn, autograd, optim
import yaml
import models.vgg16 as vg

def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

class Unpaired_model(nn.Module):
    def __init__(self, opt):
        super(Unpaired_model, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---      
        if opt.netG == "wavelet":
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            self.netS = discriminators.OASIS_Discriminator(opt)
            if opt.netDu == 'wavelet':
                self.netDu = discriminators.WaveletDiscriminator(opt)
            elif opt.netDu == 'wavelet_decoder':
                self.netDu = discriminators.WaveletDiscriminator(opt)
                self.wavelet_decoder = discriminators.Wavelet_decoder(opt)
            else :
                self.netDu = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)


    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        inv_idx = torch.arange(256 - 1, -1, -1).long().cuda()

        if mode == "losses_G":
            loss_G = 0
            fake = self.netG(label)
            output_S = self.netS(fake)
            loss_G_seg = self.opt.lambda_segment*losses_computer.loss(output_S, label, for_real=True)

            loss_G += loss_G_seg

            pred_fake = self.netDu(fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
            loss_G += loss_G_GAN

            return loss_G, [loss_G_seg, loss_G_GAN]

        if mode == "losses_S":
            loss_S = 0
            with torch.no_grad():
                fake = self.netG(label)
            output_S_fake = self.netS(fake)
            loss_S_fake = losses_computer.loss(output_S_fake, label, for_real=True)
            loss_S += loss_S_fake
            return loss_S, [loss_S_fake]

        if mode == "losses_Du":
            loss_Du = 0
            with torch.no_grad():
                fake = self.netG(label)
            output_Du_fake = self.netDu(fake)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            loss_Du += loss_Du_fake
            output_Du_real = self.netDu(image)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()
            loss_Du += loss_Du_real

            if self.opt.netDu == 'wavelet_decoder':
                losses_decoder = 0
                features = self.netDu(image, for_features=True)
                decoder_output = self.wavelet_decoder(features[0], features[1], features[2], features[3], features[4],
                                                      features[5])
                decoder_loss = nn.L1Loss()
                losses_decoder += decoder_loss(image, decoder_output).mean()
                loss_Du += losses_decoder

            return loss_Du, [loss_Du_fake,loss_Du_real]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label)
                else:
                    fake = self.netEMA(label)
            return fake

        if mode == "segment_real":
            segmentation = self.netS(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label)
            else:
                fake = self.netEMA(label)
            segmentation = self.netS(fake)
            return segmentation

        if mode == "Du_regularize":
            loss_Du = 0
            image.requires_grad = True
            real_pred = self.netDu(image)
            r1_loss = d_r1_loss(real_pred, image).mean()
            loss_Du += 10 * r1_loss
            return loss_Du, [r1_loss]

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netS.load_state_dict(torch.load(path + "S.pth"))
            self.netDu.load_state_dict(torch.load(path + "Du.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netS, self.netDu]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netS,]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'], input_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to("cuda")
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map

def tee_loss(x, y):
    return x+y, y.detach()
