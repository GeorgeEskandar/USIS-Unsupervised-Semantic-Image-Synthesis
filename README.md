# USIS Unsupervised Semantic Image Synthesis

Official PyTorch implementation of the ICASSP 2022 paper "You Only Need Adversarial Supervision for Semantic Image Synthesis" and journal paper "USIS: Unsupervised Semantic Image Synthesis". The code allows the users to reproduce and extend the results reported in the study. Please cite the paper when reporting, reproducing or extending the results.

[[IEEE ICASSP](https://ieeexplore.ieee.org/document/9746759)]  [[Arxiv](https://arxiv.org/abs/2109.14715)] [[CAG](https://www.sciencedirect.com/science/article/abs/pii/S0097849323000018)]  


# Overview

This repository implements the USIS (CAG and Arxiv) and USIS-Wavelet (ICASSP) models, which generate realistic looking images from semantic label maps in an unpaired way. This repository is built upon the supervised OASIS repository [https://github.com/boschresearch/OASIS]


<p align="center">
<img src="overview.png" >
</p>

## Setup
First, clone this repository:
```
git clone https://github.com/GeorgeEskandar/USIS-Unsupervised-Semantic-Image-Synthesis.git
cd USIS-Unsupervised-Semantic-Image-Synthesis
```

The code is tested for Python 3.7.6 and the packages listed in [oasis.yml](oasis.yml).
The basic requirements are PyTorch and Torchvision.
The easiest way to get going is to setup a conda environment via
```
conda create -n myenv gcc_linux-64=7.3 python=3.8 gxx_linux-64=7.3
conda activate myenv
conda install -c anaconda cudatoolkit=11.3
conda install -c conda-forge cudatoolkit-dev=11.3
conda install -c nvidia cudnn=7.6.5
pip install -r requirements.txt
```
Then install the pytorch-wavelet library from this repo: [https://github.com/fbcotter/pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets)

## Datasets

For COCO-Stuff, Cityscapes or ADE20K, please follow the instructions for the dataset preparation as outlined in [https://github.com/NVlabs/SPADE](https://github.com/NVlabs/SPADE).

## Training the model

To train the model, execute the usis.sh file. In this script you first need to specify the path to the data folder. Via the ```--name``` parameter the experiment can be given a unique identifier. The experimental results are then saved in the folder ```./checkpoints```, where a new folder for each run is created with the specified experiment name. You can also specify another folder for the checkpoints using the ```--checkpoints_dir``` parameter.
If you want to continue training, start the respective script with the ```--continue_train``` flag. Have a look at ```config.py``` for other options you can specify.  
Training with a batchsize of 4 is recommended for USIS and 8 for USIS-Wavelet. 
To train with the wavelet-SPADE generator, specify the option --netG wavelet, and to train with the wavelet decoder and reconstruction loss, specfiy --netDu wavelet_decoder.

## Testing the model

To test a trained model, execute the testing scripts uing the command in usis.sh file. The ```--name``` parameter should correspond to the experiment name that you want to test, and the ```--checkpoints_dir``` should the folder where the experiment is saved (default: ```./checkpoints```). These scripts will generate images from a pretrained model in ```./results/name/```.

You can generate images with a pre-trained checkpoint via ```test.py```. Using the example of ADE20K:
```
python test.py --dataset_mode ade20k --name usis_ade20k \
--dataroot path_to/ADEChallenge2016
```
This script will create a folder named ```./results``` in which the resulting images are saved.

If you want to continue training from this checkpoint, use ```train.py``` with the same ```--name``` parameter and add ```--continue_train --which_iter best```. Check the usis.sh file for examples on how to train and test the model.

## Citation
If you use this work please cite
```
@article{ESKANDAR202314,
title = {USIS: Unsupervised Semantic Image Synthesis},
journal = {Computers & Graphics},
volume = {111},
pages = {14-23},
year = {2023},
issn = {0097-8493},
doi = {https://doi.org/10.1016/j.cag.2022.12.010},
url = {https://www.sciencedirect.com/science/article/pii/S0097849323000018},
author = {George Eskandar and Mohamed Abdelsamad and Karim Armanious and Bin Yang},
}  
```
and 
```
@INPROCEEDINGS{9746759,
  author={Eskandar, George and Abdelsamad, Mohamed and Armanious, Karim and Zhang, Shuai and Yang, Bin},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Wavelet-Based Unsupervised Label-to-Image Translation}, 
  year={2022},
  volume={},
  number={},
  pages={1760-1764},
  doi={10.1109/ICASSP43922.2022.9746759}}
```

## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication cited above.

## Contact
Please feel free to open an issue or contact us personally if you have questions, need help, or need explanations.

george.eskandar@iss.uni-stuttgart.de
georgesbassem@gmail.com
