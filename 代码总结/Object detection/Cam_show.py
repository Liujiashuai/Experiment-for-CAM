"""
visual the ILSVRC dataset by explaining methods
save the result
normalize the result
"""
import numpy as np
import torch
import torchvision.models as models
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from pathlib import Path
from PIL import Image
from skimage.measure import label
import xml.etree.ElementTree as ET
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as fun
import os
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM, \
    SmoothLayerCAM

os.environ['TORCH_HOME'] = r'F:\torch_model'  # change the path to save the dataset

parser = argparse.ArgumentParser(description='The pytorch code to visual ILSVRC dataset')
parser.add_argument('--data_path', default='../../Imagenet2012/ILSVRC2012_img_val',
                    help='the data path of validation dataset')
parser.add_argument('--gt_path', default=r'E:\Alearn\interpretability\Imagenet2012\ILSVRC2012_val_bboxes',
                    help='the ground truth boxes of validation dataset')
parser.add_argument('--size', type=int, default=224, help='the size of using images')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--ex_method', default='LayerCAM', help='the methods to explain the model')
parser.add_argument('--ex_layer', default=3, help='the layer using explaining')
parser.add_argument('--save_path', default="show_result", type=str,
                    help="the path to save the result")
parser.add_argument('--model_path', type=str, default='model1')
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
    return int_area / uni_area


def get_new_box(x_min, x_max, y_min, y_max, width_, height_):
    """
    get a new box witch is cropped
    the transform process is first: resize the short side to size
    then: crop the size in the centric
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


def standardize_and_clip(img_tensor, min_value=0, max_value=1):
    std, mean = torch.std_mean(img_tensor)
    # print("the std is", std, "the mean is", mean)
    if std == 0:
        std += 1e-07
    standardized = (img_tensor - mean) / std * 0.1
    clipped = (standardized + 0.5)
    clipped[clipped > max_value] = max_value
    clipped[clipped < min_value] = min_value
    return clipped


def get_box(act_map_: torch.tensor, size_: int, threshold: float = 0.15):
    """ get the object detection boxes
    :param act_map_: activate map to explain
    :param size_: final size for act_map to resize
    :param threshold:
    :return: a boxes: [x_min, x_max, y_min, y_max]
    """
    # act_map_ = fun.interpolate(act_map_.unsqueeze(0), size=[size_, size_])
    # act_map_ = act_map_.squeeze(0).squeeze(0)
    # act_map_ = standardize_and_clip(act_map_)  # changed
    act_max = act_map_.max()
    act_bi = torch.where(act_map_ >= act_map_.min() + act_max*threshold, 1, 0)  # binary the image
    act_bi_ = act_bi.numpy()
    labeled_img, num = label(act_bi_, background=0, return_num=True, connectivity=2)
    max_num = 0
    max_label = 0
    # print("the labeled img is\n", labeled_img)
    # print("np.sum(labeled_img == num)", np.sum(labeled_img == num))
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    # print("连通区域的数量是", num, "选择的连通区域是", max_label)
    # print("np.sum(labeled_img == max label)", np.sum(labeled_img == max_label))
    act_mcr = (labeled_img == max_label)
    act_mid = np.nonzero(act_mcr)
    x_min = np.min(act_mid[1])
    x_max = np.max(act_mid[1])
    y_min = np.min(act_mid[0])
    y_max = np.max(act_mid[0])
    return [x_min, x_max, y_min, y_max]


def save_result():
    pass


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'
    # prepare the model
    model = models.vgg16(pretrained=True).eval()
    # load the model
    load_model = False
    if load_model:
        load_model_name = Path(args.model_path) / "model-9.tch"
        if device == 'cpu':
            model_check = torch.load(load_model_name, map_location='cpu')
            model.load_state_dict(model_check["state_dict"])
        else:
            model_check = torch.load(load_model_name)
            model.load_state_dict(model_check["state_dict"])
        print(f"loading model {load_model_name}")
    else:
        print("use the pretrained model")
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
    img_names = list(Path(args.data_path).glob("*"))

    # prepare the ground truth
    gt_names = list(Path(args.gt_path).glob('*'))
    img_names.sort()
    gt_names.sort()

    # def the chosen number
    chose_num = 132
    for img_name, gt_name in zip(img_names[chose_num:chose_num+1], gt_names[chose_num:chose_num+1]):
        print("==========explaining==========", img_name)
        # img = Image.open(img_name).resize((size, size)).convert('RGB')
        img = Image.open(img_name).convert('RGB')
        # img_mid = transform(img)
        img_mid = transform(img.resize((size, size)))  # changed
        img_input = Norm_transform(img_mid)
        img_input = img_input.to(device)

        # read the gt_box
        gt_tree = ET.parse(gt_name)
        gt_root = gt_tree.getroot()
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
        for ex_method in ["GradCAM", "GradCAMpp", "SmoothLayerCAM", "LayerCAM"]:
            # show the original image
            fig, axis = plt.subplots()
            gt_box = []  # ground truth boxes
            # fig.imshow(img)
            # show the gt_box
            for x1, x2, y1, y2 in zip(gt_xmin, gt_xmax, gt_ymin, gt_ymax):
                gt_box.append([x1, x2, y1, y2])
                rec = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                        linewidth=2,
                                        edgecolor='g',
                                        facecolor='none')
                axis.add_patch(rec)

            # get the explain result
            cam_extrac = get_extrac(model, ex_method=ex_method,
                                    ex_layer=args.ex_layer)
            out = model(img_input.unsqueeze(0))
            act_map = cam_extrac(out.squeeze(0).argmax().item(), out)
            act_map_ = fun.interpolate(act_map[0].unsqueeze(0), size=[size, size],
                                       mode='bilinear', align_corners=True)
            act_map_ = act_map_.squeeze(0).squeeze(0)
            act_map_ = standardize_and_clip(act_map_)  # changed
            # print("the act_map_ size is", act_map_.size())
            # print("the act_map size is", act_map[0].size())
            model.zero_grad()
            overlay_result1 = overlay_mask(img,
                                           to_pil_image(act_map_, mode='F'))
            if ex_method == "GradCAMpp":
                bboxes = get_box(act_map_, size_=size, threshold=0.15)
            elif ex_method == "GradCAM":
                bboxes = get_box(act_map_, size_=size, threshold=0.15)
            else:
                bboxes = get_box(act_map_, size_=size, threshold=0.15)
            axis.imshow(overlay_result1)

            new_bbox = get_new_box1(x_min=bboxes[0], x_max=bboxes[1],  # changed
                                    y_min=bboxes[2], y_max=bboxes[3],
                                    width_=width, height_=height)
            new_box = patches.Rectangle((new_bbox[0], new_bbox[2]),
                                        width=new_bbox[1]-new_bbox[0],
                                        height=new_bbox[3]-new_bbox[2],
                                        linewidth=2,
                                        edgecolor='r',
                                        facecolor='none')
            axis.add_patch(new_box)
            print(f"for {ex_method}, the IoU of this picture is:", cal_IoU(new_bbox, gt_boxes=gt_box))
            axis.axis('off')
            save_path = Path(args.save_path) / ex_method
            save_path.mkdir(parents=True, exist_ok=True)
            save_name = img_name.with_suffix(".png").name
            plt.savefig(save_path / f"layer-{args.ex_layer}-{save_name}")
            # plt.show()
