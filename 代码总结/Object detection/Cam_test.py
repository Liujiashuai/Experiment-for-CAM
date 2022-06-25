"""
test the whole model's IoU
by top5 and top1
for PC
"""

import numpy as np
import torch
import torchvision.models as models
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM, \
    SmoothLayerCAM
import torch.nn.functional as fun
import os
from utils import setup_seed

os.environ['TORCH_HOME'] = r'F:\torch_model'  # change the path to save the dataset
# os.environ['TORCH_HOME'] = r'torch_model'  # change the path to save the dataset
# torch.cuda.set_device(0)  # choose the GPU
setup_seed(42)

parser = argparse.ArgumentParser(description='The pytorch code to visual ILSVRC dataset')
parser.add_argument('--data_path', default='../../Imagenet2012/ILSVRC2012_img_val',
                    help='the data path of validation dataset')
parser.add_argument('--gt_path', default=r'E:\Alearn\interpretability\Imagenet2012\ILSVRC2012_val_bboxes',
                    help='the ground truth boxes of validation dataset')
parser.add_argument('--size', type=int, default=224, help='the size of using images')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--ex_method', default='GradCAMpp', type=str,
                    help='the methods to explain the model')
parser.add_argument('--ex_layer', default=5, type=int,
                    help='the layer using explaining')
args = parser.parse_args()


def get_extrac(model, ex_method, ex_layer):
    """

    :param model:
    :param ex_layer:
    :return:
    """
    if ex_layer == 4:
        layer = model.features[21]
    elif ex_layer == 3:
        layer = model.features[14]
    elif ex_layer == 2:
        layer = model.features[7]
    elif ex_layer == 1:
        layer = model.features[2]
    else:
        layer = model.features[28]

    if ex_method == "SmoothGradCAMpp":
        cam_extractor = SmoothGradCAMpp(model, target_layer=layer)
    elif ex_method == "GradCAM":
        cam_extractor = GradCAM(model, target_layer=layer)
    elif ex_method == "GradCAMpp":
        cam_extractor = GradCAMpp(model, target_layer=layer)
    elif ex_method == "XGradCAM":
        cam_extractor = XGradCAM(model, target_layer=layer)
    elif ex_method == "LayerCAM":
        cam_extractor = LayerCAM(model, target_layer=layer)
    elif ex_method == "ScoreCAM":
        cam_extractor = ScoreCAM(model, target_layer=layer)
    elif ex_method == "SSCAM":
        cam_extractor = SSCAM(model, target_layer=layer)
    elif ex_method == "ISCAM":
        cam_extractor = ISCAM(model, target_layer=layer)
    elif ex_method == "SmoothLayerCAM":
        cam_extractor = SmoothLayerCAM(model, target_layer=layer)
    else:
        print("there is no available methods.")
        return
    return cam_extractor


def cal_IoU(box, gt_boxes):
    int_area = 0  # the interaction set
    uni_area = (box[1] - box[0]) * (box[3] - box[2])  # the union set
    for gt_box in gt_boxes:
        int_width = 0
        int_height = 0
        if min(box[1], gt_box[1]) - max(box[0], gt_box[0]) > 0:
            int_width = min(box[1], gt_box[1]) - max(box[0], gt_box[0])
        if min(box[3], gt_box[3]) - max(box[2], gt_box[2]) > 0:
            int_height = min(box[3], gt_box[3]) - max(box[2], gt_box[2])
        int_area += int_width * int_height
        uni_area += (gt_box[1] - gt_box[0]) * (gt_box[3] - gt_box[2])
    uni_area -= int_area
    return int_area / uni_area, int_area, uni_area


def get_new_box(x_min, x_max, y_min, y_max, width_, height_):
    """
    get a new box witch is cropped
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param width_:
    :param height_:
    :return:
    """
    crop_size = 224
    if width_ > height_:
        new_x_min = int(round(1 / 2 * (width_ - height_) + x_min * height_ / crop_size))
        new_x_max = int(round(1 / 2 * (width_ - height_) + x_max * height_ / crop_size))
        new_y_min = int(round(y_min * height_ / crop_size))
        new_y_max = int(round(y_max * height_ / crop_size))
    else:
        new_x_min = int(round(x_min * width_ / crop_size))
        new_x_max = int(round(x_max * width_ / crop_size))
        new_y_min = int(round(1 / 2 * (height_ - width_) + y_min * width_ / crop_size))
        new_y_max = int(round(1 / 2 * (height_ - width_) + y_max * width_ / crop_size))
    return [new_x_min, new_x_max, new_y_min, new_y_max]


def get_new_box1(x_min, x_max, y_min, y_max, width_, height_):
    """
    get a new box witch is has been resized before
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param width_:
    :param height_:
    :return:
    """
    crop_size = 224
    new_x_min = int(round(x_min * width_ / crop_size))
    new_x_max = int(round(x_max * width_ / crop_size))
    new_y_min = int(round(y_min * height_ / crop_size))
    new_y_max = int(round(y_max * height_ / crop_size))
    return [new_x_min, new_x_max, new_y_min, new_y_max]


def get_box(act_map_: torch.tensor, size_: int, threshold: float = 0.4):
    """ get the object detection boxes
    :param act_map_: activate map to explain
    :param size_: final size for act_map to resize
    :param threshold:
    :return: a boxes: [x_min, x_max, y_min, y_max]
    """
    act_map_ = fun.interpolate(act_map_.unsqueeze(0), size=[size_, size_])
    act_map_ = act_map_.squeeze(0).squeeze(0)
    act_max = act_map_.max()
    act_bi = torch.where(act_map_ >= act_max*threshold, 1, 0)  # binary the image
    act_mid = torch.nonzero(act_bi)
    x_min = torch.min(act_mid[:, 1]).item()
    x_max = torch.max(act_mid[:, 1]).item()
    y_min = torch.min(act_mid[:, 0]).item()
    y_max = torch.max(act_mid[:, 0]).item()
    return [x_min, x_max, y_min, y_max]


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'
    # prepare the model
    model = models.vgg16(pretrained=True).eval()
    # print("the model is", model)
    model.to(device)

    # prepare the dataset
    size = args.size
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    transform = transforms.Compose([])
    transform.transforms.append(transforms.Resize(size))
    transform.transforms.append(transforms.CenterCrop(size))
    transform.transforms.append(transforms.ToTensor())
    Norm_transform = transforms.Compose([transforms.Normalize(means, stds)])

    # init
    int_areas = 0  # for top1
    uni_areas = 0
    int_areas_5 = 0  # for top5
    uni_areas_5 = 0
    # data_types = list(i.name for i in Path(args.data_path).glob("*"))
    # for data_type in data_types[:100]:
    img_names = list((Path(args.data_path).glob("*.JPEG")))
    # print("the images' name:", img_names)
    for img_num, img_name in enumerate(img_names[:200]):
        if img_num % 20 == 0:
            # print("+++++++++testing++++++++++", data_type)
            print(f"the total int_areas is {int_areas}, the total uni_areas is {uni_areas},"
                  f"the total int_areas_5 is {int_areas_5}, the total uni_areas_5 is {uni_areas_5},")
        # print("===========explaining==========", img_name)
        # img = Image.open(img_name).resize((size, size)).convert('RGB')
        img = Image.open(img_name).convert('RGB')
        img_mid = transform(img.resize((size, size)))  # changed
        # img_mid = transform(img)
        img_input = Norm_transform(img_mid)
        img_input = img_input.to(device)

        # find the gt_name
        gt_name = img_name.with_suffix(".xml").name
        gt_path = Path(args.gt_path) / gt_name

        # read the gt_box
        gt_tree = ET.parse(gt_path)
        gt_root = gt_tree.getroot()
        gt_box = []  # ground truth boxes
        gt_xmin = []
        gt_xmax = []
        gt_ymin = []
        gt_ymax = []
        height = 0
        width = 0
        for size_ in gt_root.iter('size'):
            for w1 in size_.iter('width'):
                width = int(w1.text)
            for h1 in size_.iter('height'):
                height = int(h1.text)
        # print('the width is', width, 'the height is', height)
        for ob in gt_root.iter('bndbox'):
            for bndbox in ob.iter('bndbox'):
                for x1 in bndbox.iter('xmin'):
                    gt_xmin.append(int(x1.text))
                for x2 in bndbox.iter('xmax'):
                    gt_xmax.append(int(x2.text))
                for y1 in bndbox.iter('ymin'):
                    gt_ymin.append(int(y1.text))
                for y2 in bndbox.iter('ymax'):
                    gt_ymax.append(int(y2.text))

        # get the gt_box
        for x1, x2, y1, y2 in zip(gt_xmin, gt_xmax, gt_ymin, gt_ymax):
            gt_box.append([x1, x2, y1, y2])

        # get the explaining box for top5 & top1

        cam_extrac = get_extrac(model, args.ex_method, args.ex_layer)
        out = model(img_input.unsqueeze(0))
        _, indexs = torch.topk(out.squeeze(0), 5)
        int_area_mid = 0
        uni_area_mid = 0
        best_IoU = 0
        for i, index in enumerate(indexs):
            if i != 0:
                out = model(img_input.unsqueeze(0))
            act_map = cam_extrac(int(index.cpu()), out)
            bboxes = get_box(act_map[0], size_=size)
            new_bbox = get_new_box1(x_min=bboxes[0], x_max=bboxes[1],
                                    y_min=bboxes[2], y_max=bboxes[3],
                                    width_=width, height_=height)
            IoU, int_area, uni_area = cal_IoU(new_bbox, gt_boxes=gt_box)
            if i == 0:
                int_areas += int_area
                uni_areas += uni_area
                # print("the top1 IoU of this picture is:", IoU)
            if IoU > best_IoU:
                best_IoU = IoU
                int_area_mid = int_area
                uni_area_mid = uni_area
        # get the total top5
        int_areas_5 += int_area_mid
        uni_areas_5 += uni_area_mid
        # print("the top5 IoU of this picture is", best_IoU)
    print('the top1 IoU of all pic is', int_areas / uni_areas)
    print('the top5 IoU of all pic is', int_areas_5 / uni_areas_5)
