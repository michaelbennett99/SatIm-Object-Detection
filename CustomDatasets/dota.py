import os
import json

from enum import Enum, auto
from zipfile import ZipFile
from pathlib import Path
from typing import Optional

from .download import download_file_from_dropbox
from .utils import listdir_nohidden, lazy_stof

import torch
import torchvision.transforms.functional as F

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torch import tensor, Tensor
from torchvision.tv_tensors._bounding_boxes import BoundingBoxes
from torchvision.utils import draw_bounding_boxes

# Download links for the DOTA dataset

DOTA_TRAIN_HBB_URL = (
    "https://www.dropbox.com/scl/fi/k5e9wdfdu4qppyz283nss/"
    "train_hbb.zip?rlkey=wrlr3fpqk8x02r5xzph8uuedk&st=kqaeuv3x&dl=1"
)
DOTA_VAL_HBB_URL = (
    "https://www.dropbox.com/scl/fi/1820a7bhat8b5esv73u6h/"
    "val_hbb.zip?rlkey=0ekae5kjspsq68cbkww8qjkl3&st=3z3ij651&dl=1"
)

DOTA_TRAIN_OBB_URL = (
    "https://www.dropbox.com/scl/fi/zu3p9wqzlu86v0kuu4v0p/"
    "train.zip?rlkey=bwih2x8xd3zpldj7l1s4owy9a&st=zrisaio3&dl=1"
)
DOTA_VAL_OBB_URL = (
    "https://www.dropbox.com/scl/fi/k3d45d22iz1op3gifazw4/"
    "val.zip?rlkey=zv7fgnkf3yqztj93cztzsfsgd&st=ghtu1ntz&dl=1"
)

# Dataset defaults

DEFAULT_DOTA_PATH = Path(os.getcwd()) / "data" / "dota"

DATASET_DICT = {
    ("train", "hbb"): {
        "url": DOTA_TRAIN_HBB_URL,
        "base_dir": os.path.join("hbb", "train")
    },
    ("val", "hbb"): {
        "url": DOTA_VAL_HBB_URL,
        "base_dir": os.path.join("hbb", "val")
    },
    ("train", "obb"): {
        "url": DOTA_TRAIN_OBB_URL,
        "base_dir": os.path.join("obb", "train")
    },
    ("val", "obb"): {
        "url": DOTA_VAL_OBB_URL,
        "base_dir": os.path.join("obb", "val")
    }
}

IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"

class Label(Enum):
    """
    Enum class for the labels in the DOTA dataset. Automatically assigns
    integers to each label.
    """
    large_vehicle = auto()
    small_vehicle = auto()
    plane = auto()
    ship = auto()
    storage_tank = auto()
    baseball_diamond = auto()
    tennis_court = auto()
    basketball_court = auto()
    ground_track_field = auto()
    harbor = auto()
    bridge = auto()
    helicopter = auto()
    roundabout = auto()
    soccer_ball_field = auto()
    swimming_pool = auto()
    container_crane = auto()
    airport = auto()
    helipad = auto()

class Target(dict):
    """
    Dict subclass that mostly acts the same but has a few extra methods for
    convenience and overwrites __str__ and __repr__ to give a more informative
    string representation.
    """
    def __init__(self, boxes = [], labels = [], difficult = [],  **kwargs):
        super().__init__(**kwargs)
        self["boxes"] = boxes
        self["labels"] = labels
        self["difficult"] = difficult

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
    """
    Class for the DOTA dataset. The dataset is split into train and val
    sets, and the annotations can be either horizontal bounding boxes (hbb)
    or oriented bounding boxes (obb).

    Once downloaded, indexing the dataset will return a tuple of an image
    and a target dictionary. The target dictionary contains the following
    keys:
        - boxes: A BoundingBoxes object containing the bounding boxes.
        - labels: A tensor containing the labels for each bounding box.
        - difficult: A tensor containing the difficulty of each bounding box.
    and any other annotation keys that are present in the target file.
    """
    def __init__(
        self,
        root: Path | str = DEFAULT_DOTA_PATH,
        split: str = "train",
        annotation_type: Optional[str] = "hbb",
        to_tensor: Optional[bool] = True,
        transforms: Optional[callable] = None,
        download: Optional[bool] = True
    ):
        """
        Initialises the DOTA dataset. Gives you either the train or val split
        with either horizontal bounding boxes (hbb) or oriented bounding boxes
        and optional conversion to tensors (required for transforms).

        Args:
            root (Path | str): The root directory to save the dataset. Defaults
                to <current working directory>/data/dota.
            split (str): The split of the dataset to use. Either "train" or
                "val". Defaults to "train".
            annotation_type (str): The type of annotation to use. Either "hbb"
                or "obb". Defaults to "hbb".
            to_tensor (bool): Whether to convert the images and targets to
                tensors. Defaults to True.
            transforms (callable): A callable transform to apply to the images
                and targets. The transform should take in an image and a target
                dictionary and return the transformed image and target. Defaults
                to None. If not none, to_tensor is set to True.
            download (bool): Whether to download the dataset if it doesn't
                exist. Defaults to True.
        """
        super().__init__(root, transforms)

        self.split = verify_str_arg(split, "split", ("train", "val"))
        self.annotation_type = verify_str_arg(
            annotation_type, "annotation_type", ("hbb", "obb")
        )

        # Transforms require the targets to be tensors
        self.to_tensor = to_tensor
        if transforms:
            self.to_tensor = True
        self.transforms = transforms

        # Cannot do transforms or tensors for obb annotations
        if self.annotation_type == "obb":
            self.to_tensor = False
            self.transforms = None

        # Set up the download
        dataset_dict = DATASET_DICT[(self.split, self.annotation_type)]

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
        target = self.parse_dota_targets(self.targets[idx])

        # Convert to tensors
        if self.to_tensor:
            img, target = self.to_tensors(img, target)

        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    @staticmethod
    def to_tensors(img: Image, target: Target) -> tuple[Tensor, Target]:
        """
        Convert images and targets to tensors.
        """
        img = F.to_tensor(img)
        target = dict(
            boxes=BoundingBoxes(
                target["boxes"],
                format="XYXY",
                canvas_size=img.shape[1:],
                dtype=img.dtype,
                device=img.device
            ),
            labels=tensor([x.value for x  in target["labels"]]),
            difficult=tensor(target["difficult"])
        )
        return img, target

    def draw_bounding_boxes(self, idx: int, width: int = 5) -> Image.Image:
        """
        Draw the bounding boxes on the image at index idx with the given width.
        """
        img, target = self.__getitem__(idx)

        # guarantee we get tensors
        if not self.to_tensor:
            img, target = self.to_tensors(img, target)

        uint8_img = (img * 255).to(torch.uint8)
        return F.to_pil_image(
            draw_bounding_boxes(uint8_img, target["boxes"], width=width)
        )

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
        dir_path = Path(path).parent
        if os.path.isfile(path) and os.path.splitext(path)[1] == ".zip":
            print(f"Unzipping {path}")
            with ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(dir_path)
            zip_dir = dir_path / listdir_nohidden(dir_path)[0]
            os.rename(zip_dir, path.removesuffix(".zip"))
            if delete:
                os.remove(path)

    @staticmethod
    def download(url: str, path: Path | str, **kwargs: Optional[dict]):
        """
        Download the zip file from dropbox and unzip it into path if path
        doesn't have the correct directory structure.
        """
        if not DOTA.val_files(path):
            download_file_from_dropbox(
                url, f"{path}.zip", **kwargs
            )

    def parse_dota_targets(self, target: str) -> Target:
        """
        Parse the DOTA target file into a Target object.
        """
        res = Target()
        with open(target, "r") as f:
            lines = f.readlines()
            for line in lines:
                ws_split = line.split()
                if len(ws_split) == 1:
                    l0, l1 = ws_split[0].split(":")
                    res.add_attribute(lazy_stof(l0), lazy_stof(l1))
                elif len(ws_split) == 10:
                    if self.annotation_type == "hbb":
                        # if annotation type is hbb, then the box is in the form
                        # [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                        box = [float(ws_split[i]) for i in [0, 1, 2, 5]]
                    else:
                        box = [float(x) for x in ws_split[:-2]]
                    label = ws_split[-2].replace("-", "_")
                    label = getattr(Label, label)
                    difficult = int(ws_split[-1])
                    res.add_box(box, label, difficult)
        return res
