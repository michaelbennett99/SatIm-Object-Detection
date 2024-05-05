import os
import json

from zipfile import ZipFile
from pathlib import Path
from typing import Optional

from download import download_file_from_dropbox
from constants import ROOT
from utils import listdir_nohidden, lazy_stof, lazy_stoi

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torch import Tensor
from torchvision.tv_tensors._bounding_boxes import BoundingBoxes

DOTA_TRAIN_URL = "xxxxxxxxxxxxx"
DOTA_VAL_URL = "xxxxxxxxxxxxx"

DEFAULT_DOTA_PATH = ROOT / "data" / "dota"

DATASET_DICT = {
    "train": {
        "url": DOTA_TRAIN_URL,
        "base_dir": "train"
    },
    "val": {
        "url": DOTA_VAL_URL,
        "base_dir": "val"
    }
}

IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"

class Target(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self["boxes"] = []
        self["labels"] = []
        self["difficult"] = []

    def __str__(self):
        return json.dumps(
            {
                "attributes": [
                    k for k in self.keys()
                    if k not in ["boxes", "labels", "difficult"]
                ],
                "features": ["boxes", "labels", "difficult"],
                "n_features": len(self['boxes'])
            }
        )

    def __repr__(self):
        return self.__str__()

    def add_attribute(self, key: str, value: str | int | float):
        self[key] = value

    def add_box(self, box: list[float], label: str, difficult: int):
        self["boxes"].append(box)
        self["labels"].append(label)
        self["difficult"].append(difficult)

class DOTA(VisionDataset):
    def __init__(
        self,
        root: Path | str = DEFAULT_DOTA_PATH,
        split: str = "train",
        transforms: Optional[callable] = None,
        download: bool = True
    ):
        super().__init__(root, transforms)

        self.split = verify_str_arg(split, "split", ("train", "val"))

        # Set up the download
        dataset_dict = DATASET_DICT[self.split]

        self.url = dataset_dict["url"]
        file_path = Path(root) / dataset_dict["base_dir"]

        if download:
            self.download(
                self.url, file_path, postprocess=self.unzip_contents
            )
        if not self.val_files(file_path):
            msg = (
                "The files in the directory are not in the correct format. ",
                "Consider setting download=True to download the files."
            )
            raise ValueError(msg)



        image_dir = file_path / IMAGES_DIRNAME
        label_dir = file_path / LABELS_DIRNAME
        self.images = sorted([
            os.path.join(image_dir, img) for img in listdir_nohidden(image_dir)
        ])
        self.targets = sorted([
            os.path.join(label_dir, tgt) for tgt in listdir_nohidden(label_dir)
        ])

        if len(self.images) != len(self.targets):
            raise ValueError(
                "Number of images and labels do not match.",
                f"Images: {len(self.images)}",
                f"Labels: {len(self.targets)}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and label
        img = Image.open(self.images[idx]).convert("RGB")
        target = self.parse_dota_labels(self.targets[idx])

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    @staticmethod
    def val_files(file_path: str) -> bool:
        """
        Make sure that the files have the following structure:

        file_path should be a directory, containing, images and labels subdirs.

        In the images subdir, every file should have the extension .png.
        There should be a .txt equivalent to each .png file in the labels
        subdir.
        """
        if not os.path.isdir(file_path):
            return False

        # Filter out hidden files like .DS_Store
        dirs = listdir_nohidden(file_path)
        if set(dirs) != set([IMAGES_DIRNAME, LABELS_DIRNAME]):
            return False

        # Check that all files in images have a corresponding file in labels
        # and the right format
        files = listdir_nohidden(file_path / IMAGES_DIRNAME)
        for file in files:
            if os.path.splitext(file)[1] != ".png":
                return False
            label = os.path.splitext(file)[0] + ".txt"
            if not os.path.isfile(file_path / LABELS_DIRNAME / label):
                return False

        return True

    @staticmethod
    def unzip_contents(path: Path | str, delete: Optional[bool] = True):
        """
        Directly unzip a single zip file and into a directory with the same name
        """
        if os.path.isfile(path) and os.path.splitext(path)[1] == ".zip":
            print(f"Unzipping {path}")
            with ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(os.path.splitext(path)[0])
            if delete:
                os.remove(path)

    @staticmethod
    def download(url: str, path: Path | str, **kwargs):
        """
        Download the zip file from dropbox and unzip it into path if path
        doesn't have the correct directory structure.
        """
        if not DOTA.val_files(path):
            download_file_from_dropbox(
                url, path, **kwargs
            )

    @staticmethod
    def parse_dota_labels(label: str) -> Target:
        res = Target()
        with open(label, "r") as f:
            lines = f.readlines()
            for line in lines:
                ws_split = line.split()
                if len(ws_split) == 1:
                    l0, l1 = ws_split[0].split(":")
                    res.add_attribute(lazy_stof(l0), lazy_stof(l1))
                elif len(ws_split) == 10:
                    box = [float(x) for x in ws_split[:8]]
                    label = ws_split[8]
                    difficult = int(ws_split[9])
                    res.add_box(box, label, difficult)
        return res
