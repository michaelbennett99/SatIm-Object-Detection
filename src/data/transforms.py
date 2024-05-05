import torch
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(300),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(180),
    v2.SanitizeBoundingBoxes(),
    v2.ToDtype(torch.float32)
])
