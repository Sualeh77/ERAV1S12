import numpy as np
import torch

from torch.utils.data import Dataset

torch.manual_seed(1)


class Cifar10Dataset(Dataset):
    """
    Dataset class for Cifar10 dataset
    """

    def __init__(self, dataset, transforms=None):
        """Initialize Dataset

        Args:
            dataset (Dataset): Pytorch Dataset instance
            transforms (Transform.Compose, optional): Tranform function instance. Defaults to None.
        """
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        """
        Get dataset length

        Returns:
            int: Length of dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Get an item form dataset

        Args:
            idx (int): id of item in dataset

        Returns:
            (tensor, int): Return tensor of transformer image, label
        """
        # Read Image and Label
        image, label = self.dataset[index]
        image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return (image.to(torch.float32), torch.tensor(label, dtype=torch.int32))
