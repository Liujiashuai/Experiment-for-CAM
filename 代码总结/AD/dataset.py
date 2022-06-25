"""
rewrite dataset;
in test mode, MVTecAT get gt(grand-truth) and test data
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed
from torchvision import transforms
from torch.utils.data import DataLoader


class Repeat(Dataset):
    """
    circulation copy the dataset, to new_length, get a new dataset
    """
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]


class MVTecAT(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, data_type, size, transform=None, gt_transform=None, mode="train"):
        """
        Args:
            root_dir (string): Directory with the MVTec AD dataset.
            data_type (string): defect to load.
            transform: Transform to apply to Data
            mode: "train" loads training samples "test" test samples default "train"
        """
        self.root_dir = Path(root_dir)
        self.data_type = data_type
        self.transform = transform
        self.gt_transform = gt_transform
        self.mode = mode
        self.size = size
        self.defect_names = list(i.name for i in (self.root_dir / data_type / mode).glob('*'))
        self.image_names = []
        self.gt_image_names = []
        if self.mode == "train":
            self.image_names = list((self.root_dir / data_type / mode / "good").glob("*.png"))
            print("loading images")
            # during training we cache the smaller images for performance reasons (not a good coding style)
            # self.imgs = [Image.open(file).resize((size,size)).convert("RGB") for file in self.image_names]
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size, size)).convert("RGB"))
                                            (file) for file in self.image_names)
            print(f"loaded {len(self.imgs)} images")
        elif self.mode == "test":
            # test mode
            for defect_name in self.defect_names:
                if defect_name == "good":
                    img_paths = list((self.root_dir / data_type / mode / "good").glob('*.png'))
                    self.image_names.extend(img_paths)
                    self.gt_image_names.extend([0] * len(img_paths))
                else:
                    img_paths = list((self.root_dir / data_type / mode / defect_name).glob('*.png'))
                    gt_img_paths = list((self.root_dir / data_type / "ground_truth" / defect_name).glob('*.png'))
                    img_paths.sort()
                    gt_img_paths.sort()
                    self.image_names.extend(img_paths)
                    self.gt_image_names.extend(gt_img_paths)
            # print("print the lens", len(self.image_names), len(self.gt_image_names))
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            # img = Image.open(self.image_names[idx])
            # img = img.convert("RGB")
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == "test":
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            # get ground truth
            gt_filename = self.gt_image_names[idx]
            if gt_filename == 0:
                gt_img = torch.zeros([1, self.size, self.size])
            else:
                gt_img = Image.open(gt_filename)
                gt_img = gt_img.resize((self.size, self.size))
                if self.gt_transform is not None:
                    gt_img = self.gt_transform(gt_img)
            return img, gt_img, label != "good"


if __name__ == "__main__":
    defect_type = 'bottle'
    size = 256

    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((size, size)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]))
    gt_transform = transforms.Compose([])
    gt_transform.transforms.append(transforms.Resize((size, size)))
    gt_transform.transforms.append(transforms.ToTensor())
    # (self, root_dir, defect_name, size, transform=None, gt_transform=None, mode="train"):
    test_dataset = MVTecAT(root_dir=r'E:\Alearn\interpretability\MvTec\mvtec_anomaly_detection', data_type='bottle',
                           size=size, transform=test_transform, gt_transform=gt_transform, mode='test')
    data_loader_test = DataLoader(test_dataset, batch_size=64,
                                  shuffle=False, num_workers=0)
    print('data loaded')
    for img, gt_img, label in data_loader_test:
        print(img.shape)
        print(torch.min(gt_img[0]), torch.max(gt_img[0]))
        print(label.shape)
