"""
version 1;
get pixel-level AUC
"""

import argparse
import torch
from density import GaussianDensitySklearn, GaussianDensityTorch
from torchvision import transforms
from dataset import MVTecAT
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
from model import ProjectionNet
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
from scipy.ndimage import gaussian_filter
from torch.nn import functional as fun
from utils import plot_roc
from collections import defaultdict
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import math


class ProjectionNet_ex(ProjectionNet):
    def __init__(self, pretrained=True, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=2):
        super(ProjectionNet_ex, self).__init__()
        # self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        self.resnet18 = resnet18(pretrained=pretrained)

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons

        # the last layer without activation

        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)

    def forward(self, x):
        embeds = self.resnet18(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return logits


def get_extractor(model, ex_method):
    """
    for explain methods, only support gradient methods
    :param model: model hold to be explained
    :param ex_method: explain method, e.g. 'GradCAM', 'GradCAM++'
    :return:
    """
    if ex_method == "SmoothGradCAMpp":
        cam_extractor = SmoothGradCAMpp(model)
    elif ex_method == "GradCAM":
        cam_extractor = GradCAM(model)
    elif ex_method == "GradCAMpp":
        cam_extractor = GradCAMpp(model)
    elif ex_method == "XGradCAM":
        cam_extractor = XGradCAM(model)
    elif ex_method == "LayerCAM":
        cam_extractor = LayerCAM(model)
    elif ex_method == "ScoreCAM":
        cam_extractor = ScoreCAM(model)
    elif ex_method == "SSCAM":
        cam_extractor = SSCAM(model)
    elif ex_method == "ISCAM":
        cam_extractor = ISCAM(model)
    else:
        print("there is no available methods.")
        return
    return cam_extractor


def run_testing(data_dir, model_name, data_type, device, model, size=256,
                head_layer=8, density=GaussianDensityTorch(), ex_method='GradCAM'):
    # create test dataset
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    gt_transform = transforms.Compose([])
    gt_transform.transforms.append(transforms.Resize((size, size)))
    gt_transform.transforms.append((transforms.ToTensor()))
    test_data_eval = MVTecAT(data_dir, data_type, size,
                             transform=test_transform, gt_transform=gt_transform, mode="test")
    dataloader_test = DataLoader(test_data_eval, batch_size=1,
                                 shuffle=False, num_workers=0)

    # Load and Create the model
    if model is None:
        head_layers = [512]*head_layer+[128]
        print(f"loading model {model_name}")
        if torch.cuda.is_available():
            model_check = torch.load(model_name)
            device = device
        else:
            model_check = torch.load(model_name, map_location=torch.device('cpu'))
            device = 'cpu'
        weights = model_check['state_dict']
        classes = weights["out.weight"].shape[0]
        model = ProjectionNet_ex(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    # get auc in pixel_level
    cam_extra = get_extractor(model, ex_method)
    gt_list_px = []  # ground truth list
    pr_list_px = []  # probability of heatmap
    for in_img, gt_img, label in dataloader_test:
        if label:
            out = model(in_img)
            activate_map = cam_extra(out.squeeze(0).argmax().item(), out)
            mask = torch.unsqueeze(activate_map[0], dim=0)
            if torch.any(torch.isnan(mask)):
                # print("==========this is error1=========\n", mask, in_img, '\n', out)
                continue
            mask = fun.interpolate(mask, size=[size, size], mode='bilinear', align_corners=True)
            if torch.any(torch.isnan(mask)):
                # print("==========this is error2=========\n", mask, in_img, '\n', out)
                continue
            # mask = gaussian_filter(mask, sigma=4)
            mask[mask > 1] = 1
            mask[mask < 0] = 0
            gt_img[gt_img >= 0.5] = 1
            gt_img[gt_img < 0.5] = 0
            # some test:
            # print("the shape of mask is:", mask.shape, "the shape of gt_img is", gt_img.shape)
            gt_list_px.extend(gt_img.cpu().numpy().astype(np.float32).ravel())
            # pr_list_px.extend(mask[0].squeeze(0).astype(np.float32).ravel())
            pr_list_px.extend(mask[0].squeeze(0).numpy().astype(np.float32).ravel())

    eval_dir = Path('eval_px') / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    print("==================test================", max(gt_list_px), max(pr_list_px), min(gt_list_px), min(pr_list_px),
          len(gt_list_px), len(pr_list_px))
    # pixel_auc = 0
    # np.savetxt('gt_list_px.txt', gt_list_px)
    # np.savetxt('pr_list_px.txt', pr_list_px)
    pixel_auc = roc_auc_score(gt_list_px, pr_list_px)
    # pixel_auc = plot_roc(gt_list_px, pr_list_px, eval_dir / "px_roc_plot.png", modelname=model_name, save_plots=True)
    return pixel_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval models for localization')
    parser.add_argument('--data_dir', default='data',
                        help='folder of input img, (default: data)')
    parser.add_argument('--model_dir', default='models',
                        help='the directory of pretrained models')
    parser.add_argument('--data_type', default='all',
                        help='the type of data type to ex, default=bottle')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--head_layer', default=2, type=int,
                        help='number of layers in the projection head (default: 8)')
    parser.add_argument('--ex_method', default='GradCAM', help='explain method, e.g. GradCAM, GradCAM++')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'

    all_types = ['bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper']
    if args.data_type == "all":
        types = all_types
    else:
        types = args.data_type.split(",")
    obj = defaultdict(list)

    # test for types
    for type in types:
        print(f"evaluating {type}")
        load_model_name = list(Path(args.model_dir).glob(f"model-{type}*"))[0]
        roc_auc = run_testing(args.data_dir, load_model_name, data_type=type, device=args.device,
                              model=None, head_layer=args.head_layer, ex_method=args.ex_method)
        print(f"{type} AUC: {roc_auc}")
        obj["defect_type"].append(type)
        obj["roc_auc"].append(roc_auc)
    df = pd.DataFrame(obj)
    df.to_csv(Path('eval_px') / "_perf.csv")
