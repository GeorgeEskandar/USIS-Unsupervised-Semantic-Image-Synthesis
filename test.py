from numpy import generic
import models.semi_supervised_models as semi_supervised_models
import models.models as models
import dataloaders.dataloaders as dataloaders
import util.utils as utils
import config
from util.fid_scores import fid_pytorch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import os
import pytorch_msssim


generate_images = True
compute_fid_generation = True

from models.generator import WaveletUpsample,InverseHaarTransform,HaarTransform,WaveletUpsample2
wavelet_upsample = WaveletUpsample()

from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
xfm = DWTForward(J=3, mode='zero', wave='db3')  # Accepts all wave types available to PyWavelets
ifm = DWTInverse(mode='zero', wave='db3')


from util.utils import tens_to_im
import numpy as np
from torch.autograd import Variable

from collections import namedtuple


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.Unpaired_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

mae = []
mse = []
len_dataloader = len(dataloader_val)
if generate_images :
    #--- iterate over validation set ---#
    for i, data_i in tqdm(enumerate(dataloader_val)):
        image, label = models.preprocess_input(opt, data_i)
        generated = model(image, label, "generate", None).cpu().detach()
        image_saver(label, generated, data_i["name"])

if compute_fid_generation :
    opt_real = opt
    opt_real.dataset_mode = "cityscapes"
    _, _, dataloader_real = dataloaders.get_dataloaders(opt)
    fid_computer = fid_pytorch(opt, dataloader_val, dataloader_real=dataloader_real , cross_domain_training=True)
    fid_computer.fid_test(model)








