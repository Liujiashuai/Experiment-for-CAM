import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
