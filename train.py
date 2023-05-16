import torch
import models.losses as losses
import models.models as models
import dataloaders.dataloaders as dataloaders
import util.utils as utils
from util.fid_scores import fid_pytorch


import config


#--- read options ---#
opt = config.read_arguments(train=True)
print("nb of gpus: ", torch.cuda.device_count())
#--- create utils ---#
timer = utils.timer(opt)
visualizer_losses = utils.losses_saver(opt)
losses_computer = losses.losses_computer(opt)
dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
im_saver = utils.image_saver(opt)
fid_computer = fid_pytorch(opt, dataloader_val)

#--- create models ---#
model = models.Unpaired_model(opt)
model = models.put_on_multi_gpus(model, opt)

#--- create optimizers ---#
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
optimizerS = torch.optim.Adam(model.module.netS.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
optimizerDu = torch.optim.Adam(model.module.netDu.parameters(), lr=5*opt.lr_d, betas=(opt.beta1, opt.beta2))

def loopy_iter(dataset):
    while True :
        for item in dataset :
            yield item

#--- the training loop ---#
already_started = False
start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))
if opt.model_supervision != 0 :
    supervised_iter = loopy_iter(dataloader_supervised)
for epoch in range(start_epoch, opt.num_epochs):
    for i, data_i in enumerate(dataloader):
        if not already_started and i < start_iter:
            continue
        already_started = True
        cur_iter = epoch*len(dataloader) + i
        image, label = models.preprocess_input(opt, data_i)

        #--- generator unconditional update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        #--- Segmentor update ---#
        model.module.netS.zero_grad()
        loss_S, losses_S_list = model(image, label, "losses_S", losses_computer)
        loss_S, losses_S_list = loss_S.mean(), [loss.mean() if loss is not None else None for loss in losses_S_list]
        loss_S.backward()
        optimizerS.step()

        #--- unconditional discriminator update ---#
        model.module.netDu.zero_grad()
        loss_Du, losses_Du_list = model(image, label, "losses_Du", losses_computer)
        loss_Du, losses_Du_list = opt.reg_every*loss_Du.mean(), [loss.mean() if loss is not None else None for loss in losses_Du_list]
        loss_Du.backward()
        optimizerDu.step()

        # --- unconditional discriminator regularize ---#
        if i % opt.reg_every == 0:
            model.module.netDu.zero_grad()
            loss_reg, losses_reg_list = model(image, label, "Du_regularize", losses_computer)
            loss_reg, losses_reg_list = loss_reg.mean(), [loss.mean() if loss is not None else None for loss in losses_reg_list]
            loss_reg.backward()
            optimizerDu.step()
        else :
            loss_reg, losses_reg_list = torch.zeros((1)), [torch.zeros((1))]

        #--- stats update ---#
        if not opt.no_EMA:
            utils.update_EMA(model, cur_iter, dataloader, opt)
        if cur_iter % opt.freq_print == 0:
            im_saver.visualize_batch(model, image, label, cur_iter)
            timer(epoch, cur_iter)
        if cur_iter % opt.freq_save_latest == 0:
            utils.save_networks(opt, cur_iter, model, latest=True)
        if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(opt, cur_iter, model, best=True)
        visualizer_losses(cur_iter, losses_G_list+losses_S_list+losses_Du_list+losses_reg_list)

#--- after training ---#
utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
utils.save_networks(opt, cur_iter, model)
utils.save_networks(opt, cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(opt, cur_iter, model, best=True)

print("The training has successfully finished")

