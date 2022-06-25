"""
version 5: compare different explain_methods
"""
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
from model import ProjectionNet
from cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
import torch
from pathlib import Path
from torchcam.utils import overlay_mask, explain_mask
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from PIL import Image
import argparse

# torch.cuda.set_device(1)  # choose the GPU


class ProjectionNet_ex(ProjectionNet):
    def __init__(self, pretrained=True, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
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


def ExplainModel(model: ProjectionNet_ex, ex_method: str, input: torch.Tensor):
    """
    for every methods to explain
    :param model:
    :param ex_method:
    :param input:
    :return:
    """
    if ex_method == "SmoothGradCAMpp":
        cam_extractor = SmoothGradCAMpp(model, target_layer=model.resnet18.layer4)
    elif ex_method == "GradCAM":
        cam_extractor = GradCAM(model, target_layer=model.resnet18.layer4)
    elif ex_method == "GradCAMpp":
        cam_extractor = GradCAMpp(model, target_layer=model.resnet18.layer4)
    elif ex_method == "XGradCAM":
        cam_extractor = XGradCAM(model, target_layer=model.resnet18.layer4)
    elif ex_method == "LayerCAM":
        cam_extractor = LayerCAM(model, target_layer=model.resnet18.layer4)
    elif ex_method == "ScoreCAM":
        cam_extractor = ScoreCAM(model, target_layer=model.resnet18.layer4)
    elif ex_method == "SSCAM":
        cam_extractor = SSCAM(model, target_layer=model.resnet18.layer4)
    elif ex_method == "ISCAM":
        cam_extractor = ISCAM(model, target_layer=model.resnet18.layer4)
    else:
        print("there is no available methods.")
        return
    out = model(input.unsqueeze(0))
    activate_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    return activate_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='explain the test_dataset')
    parser.add_argument('--data_dir', default='../MvTec/mvtec_anomaly_detection',
                        help='folder of input img, (default: data)')
    parser.add_argument('--model_dir', default='models',
                        help='the directory of pretrained models')
    parser.add_argument('--head_layer', default=4, help='num of head_layers for model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cutpaste_type', default=CutPaste3Way,
                        help='the type of cutpaste, default is CutPaste3Way')
    parser.add_argument('--data_type', default='all',
                        help='the type of data type to ex, default=bottle')
    parser.add_argument('--save_dir', default='ex_dir_compare4',
                        help='the directory to save the explain result')
    args = parser.parse_args()

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

    for type in types:
        # get the data path
        data_dir = Path(args.data_dir)
        defect_names = list(i.name for i in (data_dir / type / "test").glob('*'))

        for defect_name in defect_names:
            if defect_name == "good":
                continue
            defect_img_names = list((data_dir / type / "test" / defect_name).glob('*.png'))
            gt_img_names = list((data_dir / type / "ground_truth" / defect_name).glob('*.png'))
            defect_img_names.sort()
            gt_img_names.sort()

            for img_num in range(5):  # change num
                # creat Model:
                load_model_name = list(Path(args.model_dir).glob(f"model-{type}*"))[0]
                print(f"loading model {load_model_name}")
                if torch.cuda.is_available():
                    device = args.device
                else:
                    device = 'cpu'

                head_layers = [512] * args.head_layer + [128]
                num_classes = 2 if args.cutpaste_type is not CutPaste3Way else 3
                model = ProjectionNet_ex(pretrained=False, head_layers=head_layers, num_classes=num_classes)

                # load the model
                if device == 'cpu':
                    model_check = torch.load(load_model_name, map_location=torch.device('cpu'))
                else:
                    model_check = torch.load(load_model_name)
                model.load_state_dict(model_check['state_dict'])
                model.eval()

                # prepare the data
                data_path = defect_img_names[img_num]
                gt_path = gt_img_names[img_num]  # ground_truth picture
                print("====explaining====", data_path)
                size = 256
                img = Image.open(data_path).resize((size, size)).convert("RGB")
                gt_img = Image.open(gt_path).resize((size, size))  # ground truth
                test_transform = transforms.Compose([])
                test_transform.transforms.append(transforms.Resize((size, size)))
                test_transform.transforms.append(transforms.ToTensor())
                img_mid = test_transform(img)  # get the mid transform result
                test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]))
                img = test_transform(img)

                # Methods: SmoothGradCAMpp, GradCAM, GradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM
                activate_map1 = ExplainModel(model, 'GradCAM', input=img)
                activate_map2 = ExplainModel(model, 'GradCAMpp', input=img)
                activate_map3 = ExplainModel(model, 'LayerCAM', input=img)
                # activate_map4 = ExplainModel(model, 'XGradCAMpp', input=img)
                # show the result, take img_mid to show
                show_act = False
                alpha = 0.8
                # plt.imshow(np.array(img1.permute(1, 2, 0)))
                figs, axis = plt.subplots(1, 6)
                # show the result
                # the original image
                axis[0].imshow(img_mid.permute(1, 2, 0))
                axis[0].axis('off')

                # the ground-truth image
                axis[1].imshow(gt_img)
                axis[1].axis('off')

                # overlay it on input image
                overlay_result1 = overlay_mask(to_pil_image(img_mid),
                                               to_pil_image(activate_map1[0].squeeze(0), mode='F'), alpha=alpha)
                axis[2].imshow(overlay_result1)
                axis[2].axis('off')

                # overlay it on input image
                overlay_result2 = overlay_mask(to_pil_image(img_mid),
                                               to_pil_image(activate_map2[0].squeeze(0), mode='F'), alpha=alpha)
                axis[3].imshow(overlay_result2)
                axis[3].axis('off')

                # LayerCAM
                overlay_result3 = overlay_mask(to_pil_image(img_mid),
                                               to_pil_image(activate_map3[0].squeeze(0), mode='F'), alpha=alpha)
                axis[4].imshow(overlay_result3)
                axis[4].axis('off')

                # XGradCAM
                # overlay_result4 = overlay_mask(to_pil_image(img_mid),
                #                                to_pil_image(activate_map4[0].squeeze(0), mode='F'), alpha=alpha)
                # axis[5].imshow(overlay_result4)
                # axis[5].axis('off')

                plt.tight_layout()
                plt.show()
                # # save the explain result
                # save_dir = Path(args.save_dir)
                # save_dir.mkdir(parents=True, exist_ok=True)
                # plt.savefig(save_dir / f"{type}-{defect_name}-{img_num}.png")
                # plt.close()
