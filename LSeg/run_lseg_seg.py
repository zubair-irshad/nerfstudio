import os

import torch

torch.cuda.device_count()

import argparse
import copy
import functools
import glob
import itertools
import math
import os
import types
from collections import OrderedDict

import clip
import encoding.utils as utils
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transform
import torchvision.transforms as torch_transforms
import torchvision.transforms as transforms
from additional_utils.models import LSeg_MultiEvalModule
from data import get_dataset
from encoding.datasets import test_batchify_fn
from encoding.models.sseg import BaseNet
from encoding.nn import SegmentationLosses, SyncBatchNorm
from encoding.parallel import DataParallelCriterion, DataParallelModel
from matplotlib.backends.backend_agg import FigureCanvasAgg
from modules.lseg_module import LSegModule
from PIL import Image
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument("--model", type=str, default="encnet", help="model name (default: encnet)")
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
        parser.add_argument("--workers", type=int, default=16, metavar="N", help="dataloader threads")
        parser.add_argument("--base-size", type=int, default=520, help="base image size")
        parser.add_argument("--crop-size", type=int, default=480, help="crop image size")
        parser.add_argument(
            "--train-split",
            type=str,
            default="train",
            help="dataset train split (default: train)",
        )
        parser.add_argument("--aux", action="store_true", default=False, help="Auxilary Loss")
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument("--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)")
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
        parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
        # checking point
        parser.add_argument("--weights", type=str, default="", help="checkpoint to test")
        # evaluation option
        parser.add_argument("--eval", action="store_true", default=False, help="evaluating mIoU")
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
            default="lseg",
            help="select model definition",
        )

        # test option
        parser.add_argument("--data-path", type=str, default="../datasets/", help="path to test image folder")

        parser.add_argument(
            "--no-scaleinv",
            dest="scale_inv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument("--widehead", default=False, action="store_true", help="wider output head")

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
            choices=["lrelu", "tanh"],
            default="lrelu",
            help="use which activation to activate the block",
        )

        parser.add_argument("--exp_id", type=int)

        parser.add_argument("--room_name", type=str)

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        print(args)
        return args


args = Options().parse()

torch.manual_seed(args.seed)
args.test_batch_size = 1
alpha = 0.5

args.scale_inv = False
args.widehead = True
args.dataset = "ade20k"
args.backbone = "clip_vitl16_384"
args.weights = "checkpoints/demo_e200.ckpt"
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
    activation="lrelu",
)

input_transform = module.val_transform

# dataloader
loader_kwargs = {"num_workers": args.workers, "pin_memory": True} if args.cuda else {}

# model
if isinstance(module.net, BaseNet):
    model = module.net
else:
    model = module

model = model.eval()
model = model.cpu()
scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == "citys" else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

model.mean = [0.5, 0.5, 0.5]
model.std = [0.5, 0.5, 0.5]

evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=False).cuda()
evaluator.eval()

from tqdm import tqdm

images = []
features = []
imgdir = "/data/zubair/replica/room_0/rgb"

out_img_feat_dir = "/data/zubair/replica/room_0/img_feat"

import os

os.makedirs(out_img_feat_dir, exist_ok=True)

# img_root = imgdir + '/' + args.room_name
# img_paths = [os.path.join(img_root, path) for path in os.listdir(img_root) if '.png' in path]


# img_paths = [os.path.join(imgdir, path) for path in os.listdir(imgdir) if ".png" in path]

img_paths = os.listdir(imgdir)

train_ids = list(range(0, len(img_paths), 5))
test_ids = [x + 2 for x in train_ids]

print("train_ids", train_ids)
print("test_ids", test_ids)

# rgb_path is rgb_0, rgb_1
for k, img_name in enumerate(tqdm(img_paths)):
    # print(k, img_name)

    # I want to check if the number associated with image_name is in train or test ids

    num = int(img_name.split("_")[1].split(".")[0])

    if num not in train_ids and num not in test_ids:
        continue

    # if k not in test_ids or k not in train_ids:
    #     continue
    img_path = os.path.join(imgdir, img_name)
    pil_img = Image.open(img_path)
    image = np.array(pil_img)[..., :3]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((int(image.shape[0]), int(image.shape[1]))),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    image = transform(image).unsqueeze(0).cuda()

    label_src = "road"

    labels = []

    lines = label_src.split(",")
    for line in lines:
        label = line
        labels.append(label)

    oo = len(img_path) - 1
    while img_path[oo] != "/":
        oo -= 1

    with torch.no_grad():
        features, text_features, outputs = evaluator.parallel_forward(
            image, labels
        )  # evaluator.forward(image, labels) #parallel_forward

    print("outputs", outputs[0].shape)
    np.savez(
        os.path.join(out_img_feat_dir, img_name.replace(".png", ".npy")),
        features[0].squeeze().cpu().numpy().astype(np.float16),
    )
