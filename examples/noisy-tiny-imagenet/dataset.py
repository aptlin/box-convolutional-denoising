import os
import glob
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image

EXTENSION = "JPEG"
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = "wnids.txt"
VAL_ANNOTATION_FILE = "val_annotations.txt"


# Adapted from https://git.io/JvkmH
class NoisyTinyImagenet(Dataset):
    """
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G)
        and want to minimize disk IO overhead. """

    def __init__(
        self,
        root,
        split="train",
        in_memory=False,
        transform=None,
        noise_mean=0,
        noise_std=1,
    ):
        self.root = os.path.expanduser(root)
        self.split = split
        self.in_memory = in_memory
        self.transform = transform
        self.split_dir = os.path.join(root, self.split)

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.image_paths = sorted(
            glob.iglob(
                os.path.join(self.split_dir, "**", "*.{}".format(EXTENSION)),
                recursive=True,
            )
        )
        self.images = []  # used for in-memory processing
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        noise = self.noise_mean + self.noise_std * torch.randn_like(img)
        return img + noise, noise

    def read_image(self, path):
        img = Image.open(path)
        if not self.transform:
            self.transform = transforms.ToTensor()
        return self.transform(img)
