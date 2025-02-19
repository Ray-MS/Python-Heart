"""
Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?
https://doi.org/10.1109/TMI.2018.2837502
https://www.creatis.insa-lyon.fr/Challenge/acdc/
"""

import re
from pathlib import Path

import monai.data
import monai.transforms as MT
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from einops import rearrange

from datasets._factory import register_dataset
from datasets.heart.dataset import HeartDataset, IndexInfo

img_mean, img_std = 72.60145, 70.6021
info_mean = torch.tensor((170.82794, 76.33533, 25.713934, 26.017376))
info_std = torch.tensor((9.745075, 18.573421, 6.64516,  5.546235))


def load_cfg(fpath: Path) -> dict:
    cfg = dict()
    text = Path.read_text(fpath)
    for key in ('Height', 'Weight', 'NbFrame',):
        cfg[key] = float(re.findall(f'{key}: ((?:\d+\.\d+|\d+))', text)[0])
    cfg['bmi'] = cfg['Weight'] / (cfg['Height']/100)**2
    return cfg


class ACDC(HeartDataset):
    def __init__(
        self,
        root: str = './data',
        img_size: int = 224,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root, 'ACDC').expanduser()
        self.img_size = img_size
        self.train = train

    def _create_index(self) -> dict[str, IndexInfo]:
        index_info = dict()

        for folder in ('training', 'testing',):
            for patient in Path.iterdir(self.raw_folder/folder):
                info = IndexInfo()

                info.fpath = str(patient.relative_to(self.raw_folder))
                cfg = Path.read_text(patient/'Info.cfg')

                for tag in ('ED', 'ES',):
                    frame = int(re.findall(f'{tag}: (\\d+)', cfg)[0])
                    seg_path = patient/f'{patient.name}_frame{frame:02d}_gt.nii.gz'
                    seg = nib.load(seg_path)
                    seg = seg.get_fdata().astype(np.uint8)
                    seg = rearrange(seg, 'h w c -> c h w')
                    setattr(info, tag, [c for c, slice in enumerate(seg) if slice.max() > 0])

                index_info[patient.name] = info

        return index_info

    def __len__(self) -> int:
        return sum(info.n_slice for _, info in self.index_info.items())

    def __getitem__(self, index: int) -> ...:
        patient, info, index = self._find_entry_by_index(index)
        folder = self.raw_folder/info.fpath

        tag = 'ED' if index < len(info.ED) else 'ES'
        cfg_path = folder/'Info.cfg'
        cfg_text = Path.read_text(cfg_path)
        frame = int(re.findall(f'{tag}: (\\d+)', cfg_text)[0])
        index = index if index < len(info.ED) else index-len(info.ED)
        cfg = load_cfg(cfg_path)
        cfg = torch.tensor(list(cfg.values()))
        cfg = cfg.__sub__(info_mean).__div__(info_std)
        cfg = rearrange(cfg, 'l -> l 1')

        img_path = folder/f'{patient}_frame{frame:02d}.nii.gz'
        img = nib.load(img_path)
        pixdim = img.header['pixdim'][1:3]
        img = img.get_fdata().astype(np.float32)[:, :, index]
        img = ndi.zoom(img, pixdim, order=3)
        img = rearrange(img, 'h w -> 1 h w')

        seg_path = folder/f'{patient}_frame{frame:02d}_gt.nii.gz'
        seg = nib.load(seg_path)
        pixdim = seg.header['pixdim'][1:3]
        seg = seg.get_fdata().astype(np.uint8)[:, :, index]
        seg = ndi.zoom(seg, pixdim, order=0)
        seg = rearrange(seg, 'h w -> 1 h w')

        data = dict()
        data['img'] = torch.from_numpy(img).__sub__(img_mean).__div__(img_std)
        data['seg'] = torch.from_numpy(seg)

        both = ('img', 'seg',)
        transform = MT.Compose([
            MT.ResizeWithPadOrCropD(both, int(1.2*self.img_size)),
            MT.RandSpatialCropD(both, self.img_size) if self.train else MT.CenterSpatialCropD(both, self.img_size),
        ])
        data = transform(data)

        return data['img'], data['seg'], cfg


@register_dataset
def acdc(root: str, **kwargs) -> tuple[ACDC, dict]:
    dataset = ACDC(root, **kwargs)
    cfg = dict()
    return dataset, cfg
