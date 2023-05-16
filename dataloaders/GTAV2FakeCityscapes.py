import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image

gta_root = r"/data/public/gta"

class GTAV2FakeCityscapes(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics, for_supervision = False):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 34
        opt.contain_dontcare_label = True
        opt.semantic_nc = 35 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.paths = self.list_images()


    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        image, label = self.transforms(image, label)
        label = label * 255

        return {"image": image, "label": label, "name": self.images[idx]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"

        if mode == "train":
            images = []
            path_img = os.path.join(r"/no_backups/m142/vsait/images_gta")
            # path_img = os.path.join(r"/data/public/mapillary/training/images")
            for item in sorted(os.listdir(path_img)):
                if int(item.split(".")[0]) <= 20000:
                    images.append(os.path.join(path_img, item))
            labels = []
            path_lab = os.path.join(gta_root, "labels")
            # path_lab = os.path.join("/no_backups/s1422/output_labels/training/id")
            for item in sorted(os.listdir(path_lab)):
                if int(item.split(".")[0]) <= 20000:
                    labels.append(os.path.join(path_lab, item))
        else:

            images = []
            path_img = os.path.join(r"/no_backups/m142/vsait/images_gta")
            # path_img = os.path.join(r"/data/public/mapillary/training/images")
            for item in sorted(os.listdir(path_img)):
                if int(item.split(".")[0]) > 20000:
                    images.append(os.path.join(path_img, item))
            labels = []
            path_lab = os.path.join(gta_root, "labels")
            # path_lab = os.path.join("/no_backups/s1422/output_labels/training/id")
            for item in sorted(os.listdir(path_lab)):
                if int(item.split(".")[0]) > 20000:
                    labels.append(os.path.join(path_lab, item))


        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        return images, labels, (path_img, path_lab)

    def transforms(self, image, label):
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label