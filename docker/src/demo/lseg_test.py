import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
torch.cuda.device_count()
import sys

import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transform

sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/utils')
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/lseg/modules')
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/lseg/modules/models')
sys.path.insert(0, '/workspaces/ROB-8/docker/src/content/vlmaps/lseg/additional_utils')

from lseg_net import LSegEncNet
from models import LSeg_MultiEvalModule
from lsegmentation_module import LSegmentationModule

class LSegModule(LSegmentationModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super(LSegModule, self).__init__(data_path, dataset, batch_size, base_lr, max_epochs, **kwargs)

        if dataset == "citys":
            self.base_size = 2048
            self.crop_size = 768
        else:
            self.base_size = 520
            self.crop_size = 480

        use_pretrained = True
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        print("** Use norm {}, {} as the mean and std **".format(norm_mean, norm_std))

        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

        # self.trainset = self.get_trainset(
        #     dataset,
        #     augment=kwargs["augment"],
        #     base_size=self.base_size,
        #     crop_size=self.crop_size,
        # )

        # self.valset = self.get_valset(
        #     dataset,
        #     augment=kwargs["augment"],
        #     base_size=self.base_size,
        #     crop_size=self.crop_size,
        # )

        use_batchnorm = (not kwargs["no_batchnorm"]) if "no_batchnorm" in kwargs else True
        # print(kwargs)

        labels = self.get_labels("ade20k")

        self.net = LSegNet(
            labels=labels,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=self.crop_size,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
        )

        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        self._up_kwargs = up_kwargs
        self.mean = norm_mean
        self.std = norm_std

        # self.criterion = self.get_criterion(**kwargs)

    def get_labels(self, dataset):
        labels = []
        path = "label_files/{}_objectInfo150.txt".format(dataset)
        assert os.path.exists(path), "*** Error : {} not exist !!!".format(path)
        f = open(path, "r")
        lines = f.readlines()
        for line in lines:
            label = line.strip().split(",")[-1].split(";")[0]
            labels.append(label)
        f.close()
        if dataset in ["ade20k"]:
            labels = labels[1:]
        return labels

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone network",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of featurs that go from encoder to decoder",
        )

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        parser.add_argument("--finetune_weights", type=str, help="load weights to finetune from")

        parser.add_argument(
            "--no-scaleinv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--no-batchnorm",
            default=False,
            action="store_true",
            help="turn off batchnorm",
        )

        parser.add_argument("--widehead", default=False, action="store_true", help="wider output head")

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=["lrelu", "tanh"],
            default="lrelu",
            help="use which activation to activate the block",
        )

        return parser

import math
import types
import functools
import torchvision.transforms as torch_transforms
import copy
import itertools
from PIL import Image
import matplotlib.pyplot as plt
import clip
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torchvision.transforms as transforms

class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument(
            "--model", type=str, default="encnet", help="model name (default: encnet)"
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone name (default: resnet50)",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ade20k",
            help="dataset name (default: pascal12)",
        )
        parser.add_argument(
            "--workers", type=int, default=16, metavar="N", help="dataloader threads"
        )
        parser.add_argument(
            "--base-size", type=int, default=520, help="base image size"
        )
        parser.add_argument(
            "--crop-size", type=int, default=480, help="crop image size"
        )
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            training (default: auto)",
        )
        parser.add_argument(
            "--test-batch-size",
            type=int,
            default=16,
            metavar="N",
            help="input batch size for \
                            testing (default: same as batch size)",
        )
        # cuda, seed and logging
        parser.add_argument(
            "--no-cuda",
            action="store_true",
            default=False,
            help="disables CUDA training",
        )
        parser.add_argument(
            "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
        )
        # checking point
        parser.add_argument(
            "--weights", type=str, default='', help="checkpoint to test"
        )
        # evaluation option
        parser.add_argument(
            "--eval", action="store_true", default=False, help="evaluating mIoU"
        )
        parser.add_argument(
            "--export",
            type=str,
            default=None,
            help="put the path to resuming file if needed",
        )
        parser.add_argument(
            "--acc-bn",
            action="store_true",
            default=False,
            help="Re-accumulate BN statistics",
        )
        parser.add_argument(
            "--test-val",
            action="store_true",
            default=False,
            help="generate masks on val set",
        )
        parser.add_argument(
            "--no-val",
            action="store_true",
            default=False,
            help="skip validation during training",
        )

        parser.add_argument(
            "--module",
            default='lseg',
            help="select model definition",
        )

        # test option
        parser.add_argument(
            "--data-path", type=str, default='../datasets/', help="path to test image folder"
        )

        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )
        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        
        parser.add_argument(
            "--label_src",
            type=str,
            default="default",
            help="how to get the labels",
        )
        
        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args(args=[]) 
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args
    

def get_new_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete

def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches

args = Options().parse()

torch.manual_seed(args.seed)
args.test_batch_size = 1 
alpha=0.5
    
args.scale_inv = False
args.widehead = True
args.dataset = 'ade20k'
args.backbone = 'clip_vitl16_384'
args.weights = 'checkpoints/demo_e200.ckpt'
args.ignore_index = 255

module = LSegModule.load_from_checkpoint(
    checkpoint_path=args.weights,
    data_path=args.data_path,
    dataset=args.dataset,
    backbone=args.backbone,
    aux=args.aux,
    num_features=256,
    aux_weight=0,
    se_loss=False,
    se_weight=0,
    base_lr=0,
    batch_size=1,
    max_epochs=0,
    ignore_index=args.ignore_index,
    dropout=0.0,
    scale_inv=args.scale_inv,
    augment=False,
    no_batchnorm=False,
    widehead=args.widehead,
    widehead_hr=args.widehead_hr,
    map_locatin="cpu",
    arch_option=0,
    block_depth=0,
    activation='lrelu',
)

input_transform = module.val_transform

# dataloader
loader_kwargs = (
    {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}
)


model = module
    
model = model.eval()
model = model.cpu()
scales = (
    [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    if args.dataset == "citys"
    else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
)  

model.mean = [0.5, 0.5, 0.5]
model.std = [0.5, 0.5, 0.5]

img_path = 'inputs/cat1.jpeg'
#img_path = 'inputs/catdog.png'
crop_size = 480
padding = [0.0] * 3
image = Image.open(img_path)
image = np.array(image)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
image = transform(image).unsqueeze(0)
img = image[0].permute(1,2,0)
img = img * 0.5 + 0.5
plt.imshow(img)

args.label_src = 'plant,grass,cat,stone,other'

labels = []
print('** Input label value: {} **'.format(args.label_src))
lines = args.label_src.split(',')
for line in lines:
    label = line
    labels.append(label)

with torch.no_grad():
    outputs = evaluator.parallel_forward(image, labels) #evaluator.forward(image, labels) #parallel_forward
    #outputs = model(image,labels)
    predicts = [
        torch.max(output, 1)[1].cpu().numpy() 
        for output in outputs
    ]
    
predict = predicts[0]

# show results
new_palette = get_new_pallete(len(labels))
mask, patches = get_new_mask_pallete(predict, new_palette, out_label_flag=True, labels=labels)
img = image[0].permute(1,2,0)
img = img * 0.5 + 0.5
img = Image.fromarray(np.uint8(255*img)).convert("RGBA")
seg = mask.convert("RGBA")
out = Image.blend(img, seg, alpha)
plt.axis('off')
plt.imshow(img)
plt.figure()
plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 1), prop={'size': 20})
plt.axis('off')
plt.imshow(seg)