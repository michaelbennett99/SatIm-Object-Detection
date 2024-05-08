import os

from pathlib import Path
from shutil import copytree
from typing import Optional

ORIG_NAMES = {
    'plane': 0,
    'ship': 1,
    'storage tank': 2,
    'baseball diamond': 3,
    'tennis court': 4,
    'basketball court': 5,
    'ground track field': 6,
    'harbor': 7,
    'bridge': 8,
    'large vehicle': 9,
    'small vehicle': 10,
    'helicopter': 11,
    'roundabout': 12,
    'soccer ball field': 13,
    'swimming pool': 14
}

NEW_NAMES = {
    'bridge': 0,
    'small vehicle': 1,
    'large vehicle': 2,
    'ship': 3,
    'plane': 4,
    'harbor': 5
}

MAPPING = {ORIG_NAMES[k]: NEW_NAMES[k] for k in NEW_NAMES.keys()}

def modify_labels(path: Path | str, copy_path: Optional[Path | str] = None):
    yaml_path = f"{path}.yaml"
    with open(yaml_path, 'r') as f:
        lines = f.readlines()

    if copy_path:
        copytree(path, copy_path)
        path = copy_path
        yaml_path = f"{copy_path}.yaml"

    with open(yaml_path, 'w') as f:
        i = 0
        while lines[i] != '\n':
            f.write(lines[i])
            i += 1
        f.write('\n')
        f.write('names:\n')
        for name, name_val in NEW_NAMES.items():
            f.write(f"    {name_val}: {name}\n")

    print("Stripping annotations...")
    n_stripped = 0
    images_dir = Path(path) / 'images'
    labels_dir = Path(path) / 'labels'
    for split in ['train', 'val']:
        for label in os.scandir(labels_dir / split):
            valid_boxes = 0
            with open(label, 'r') as f:
                lines = f.readlines()
            with open(label, 'w') as f:
                valid_boxes = 0
                for line in lines:
                    cat, res = line.split(" ", 1)
                    cat = int(cat)
                    if cat in MAPPING:
                        f.write(f"{MAPPING[cat]} {res}")
                        valid_boxes += 1
            if valid_boxes == 0:
                os.remove(label)
                os.remove(images_dir / split / label.name.replace('txt', 'jpg'))
                n_stripped += 1
    print(
        f"Finished. {n_stripped} images with no valid annotations were removed."
    )
