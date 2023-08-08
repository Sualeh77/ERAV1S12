import torch
import albumentations as A
from albumentations.augmentations.geometric.transforms import PadIfNeeded
from albumentations.pytorch import ToTensorV2

from torchvision import transforms

torch.manual_seed(1)

def get_a_train_transform():
    """
    Get transformer for training data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize((0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Sequential([
            PadIfNeeded(min_height=40, min_width=40, always_apply=True),
            A.RandomCrop(32,32)
        ]),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,
                         fill_value=((0.4914, 0.4822, 0.4465)), mask_fill_value = None),
        A.Resize(32,32),
        ToTensorV2()
    ])

def get_a_test_transform():
    """
    Get transformer for test data

    Returns:
        Compose: Composed transformations
    """
    return A.Compose([
        A.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
        ToTensorV2()
    ])

def get_p_train_transform():
    """Get Pytorch Transform function for train data

    Returns:
        Compose: Composed transformations
    """
    random_rotation_degree = 5
    img_size = (32, 32)
    random_crop_percent = (0.85, 1.0)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, random_crop_percent),
        transforms.RandomRotation(random_rotation_degree),
        transforms.ToTensor(),
        transforms.Normalize()
    ])

def get_p_test_transform():
    """
    Get Pytorch Transform function for test data

    Returns:
        Compose: Composed transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
    ])