#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=usis
#SBATCH --output=usis%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4-23:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1

# Activate everything you need

conda activate /anaconda3/envs/myenv
# Run your python code

python train.py --name usis_wavelet --dataset_mode cityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes --batch_size 1  \
--netDu wavelet \
--model_supervision 0 --netG wavelet --channels_G 16

#python train.py --name usis --dataset_mode cityscapes --gpu_ids 0 \
#--dataroot /data/public/cityscapes --batch_size 1  \
#--netDu wavelet_decoder \
#--model_supervision 0 --netG 0 --channels_G 16

#python test.py --name oasis_cityscapes_wavelet_disc --dataset_mode cityscapes --gpu_ids 0 \
#--dataroot /data/public/cityscapes --batch_size 1 \
#--channels_G 64 --netG 0 \
#--model_supervision 0 \
#--ckpt_iter best



