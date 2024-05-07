import os
import json

from pathlib import Path
from typing import Optional
from warnings import warn

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg, extract_archive

from .download import download_file_from_gdrive_gdown, unzip_file
from .utils import listdir_nohidden

VALID_CATEGORY_URL = "https://drive.google.com/uc?id=1Av9tHAamg2oH_JpOpqNH1Db9C5TmKfH2"
VALID_LABELS_URL = "https://drive.google.com/uc?id=1F6Xp5QLmE9vwjwzh82Gsq1jUyzV8m7ds"
VALID_IMAGES_URL = "https://drive.google.com/uc?id=1Q1j_OgNlnJG2RlzQniFSIpLjozGAJ1D_"

# Dataset defaults
DEFAULT_VALID_PATH = Path(os.getcwd()) / "data" / "valid"

DATASET_DICT = {
    "category": {
        "url": VALID_CATEGORY_URL,
        "path": "category.json",
        "name": "category.json"
    },
    "labels": {
        "url": VALID_LABELS_URL,
        "path": "labels.zip",
        "name": "label"
    },
    "images": {
        "url": VALID_IMAGES_URL,
        "path": "images.zip",
        "name": "images"
    }
}

def extract(
    path: Path | str,
):
    if path.endswith(".zip"):
        unzip_file(path, remove_finished=True)

class VALID(VisionDataset):
    def __init__(
        self,
        root: Optional[Path | str] = DEFAULT_VALID_PATH,
        annotation_type: Optional[str] = "hbb",
        to_tensor: Optional[bool] = True,
        transforms: Optional[callable] = None,
        download: Optional[bool] = True
    ) -> None:
        super().__init__(root, transforms)

        self.annotation_type = verify_str_arg(
            annotation_type, "annotation_type", ("hbb", "obb")
        )

        # transforms requre targets to be tensores
        self.to_tensor = to_tensor
        if transforms:
            self.to_tensor = True
        self.transforms = transforms

        self.dataset_dict = DATASET_DICT
        self.subdirs = [x['name'] for x in self.dataset_dict.values()]

        # cannot do transforms or tensors for obb annotations
        if self.annotation_type == "obb":
            msg = (
                "OBB annotations are not supported yet. This is an issue with "
                "pytorch's torchvision library. Please use HBB annotations. "
                "OBB dataloader will return images and targets in native "
                "format that is not compatible with torchvision."
            )
            warn(msg)
            self.to_tensor = False
            self.transforms = None

        # set up the download
        if download:
            self.download()


    # TODO: Check download still works with self.root
    def download(self):
        for _, value in self.dataset_dict.items():
            download_file_from_gdrive_gdown(
                value["url"], os.path.join(self.root, value['path']),
                postprocess=extract
            )

    def val_files(self):
        if not os.path.isdir(self.root):
            return False

        contents = listdir_nohidden(self.root)
        if set(contents) != set(self.subdirs):
            return False

        # check that all files in images have a corresponding label
        labels_dir = os.path.join(
            self.root, self.dataset_dict["labels"]["name"]
        )
        labels = listdir_nohidden(labels_dir)
        for label in labels:
            with open(os.path.join(labels_dir, label)) as f:
                file_name = json.load(f)['file_name']
            if not os.path.isfile(os.path.join(self.root, file_name)):
                return False

        return True
