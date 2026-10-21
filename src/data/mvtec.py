from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CATEGORIES = [
    "carpet", "grid", "leather", "tile", "wood",
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]

class MVTecDataset(Dataset):
    def __init__(self, root, category, split="train", img_size=256, crop_size=224):
        assert split in ("train", "test")
        self.root = Path(root) / category
        self.split = split
        self.crop_size = crop_size

        self.img_tf = T.Compose([
            T.Resize(img_size, T.InterpolationMode.LANCZOS),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mask_tf = T.Compose([
            T.Resize(img_size, T.InterpolationMode.NEAREST),
            T.CenterCrop(crop_size),
            T.ToTensor(),
        ])
        self.samples = self._collect()

    def _collect(self):
        samples = []  # (img_path, mask_path or None, label, defect_type)
        if self.split == "train":
            for p in sorted((self.root / "train" / "good").glob("*.png")):
                samples.append((p, None, 0, "good"))
        else:
            for d in sorted((self.root / "test").iterdir()):
                if not d.is_dir():
                    continue
                for p in sorted(d.glob("*.png")):
                    if d.name == "good":
                        samples.append((p, None, 0, "good"))
                    else:
                        m = self.root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                        samples.append((p, m, 1, d.name))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_p, mask_p, label, dtype = self.samples[i]
        img = self.img_tf(Image.open(img_p).convert("RGB"))
        if mask_p is not None:
            mask = (self.mask_tf(Image.open(mask_p).convert("L")) > 0.5).float()
        else:
            mask = torch.zeros(1, self.crop_size, self.crop_size)
        return dict(image=img, mask=mask, label=label, defect_type=dtype, path=str(img_p))
