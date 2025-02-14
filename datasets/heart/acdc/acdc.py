"""
Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?
https://doi.org/10.1109/TMI.2018.2837502
https://www.creatis.insa-lyon.fr/Challenge/acdc/
"""

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from einops import rearrange

from datasets._factory import register_dataset
from datasets.heart.dataset import HeartDataset, IndexInfo


class ACDC(HeartDataset):
    def __init__(
        self,
        root: str = './data',
    ) -> None:
        super().__init__()
        self.root = Path(root, 'ACDC').expanduser()

    def _create_index(self) -> None:
        index_info = dict()

        for idx in range(1, 151):
            patient = f'patient{idx:03d}'
            info = IndexInfo()

            info.fpath = f'training/{patient}' if idx <= 100 else f'testing/{patient}'
            folder = self.raw_folder/info.fpath
            cfg = Path.read_text(folder/'Info.cfg')

            for tag in ('ED', 'ES'):
                frame = int(re.findall(f'{tag}: (\\d+)', cfg)[0])
                seg_path = folder/f'{patient}_frame{frame:02d}_gt.nii.gz'
                seg = nib.load(seg_path)
                seg = seg.get_fdata().astype(np.uint8)
                seg = rearrange(seg, 'h w c -> c h w')
                setattr(info, tag, [c for c, slice in enumerate(seg) if slice.max() > 0])

            index_info[patient] = info

        return index_info

    def __len__(self) -> int:
        return sum(info.n_slice for _, info in self.index_info.items())

    def __getitem__(self, index: int) -> ...:
        patient, info, index = self._find_entry_by_index(index)
        cls = 'ED' if index < len(info.ED) else 'ES'

        folder = self.raw_folder/info.fpath
        cfg = Path.read_text(folder/'Info.cfg')
        frame = int(re.findall(f'{cls}: (\\d+)', cfg)[0])
        index = index if index < len(info.ED) else index-len(info.ED)

        img_path = folder/f'{patient}_frame{frame:02d}.nii.gz'
        img = nib.load(img_path)
        img = img.get_fdata().astype(np.float32)[:, :, index]

        seg_path = folder/f'{patient}_frame{frame:02d}_gt.nii.gz'
        seg = nib.load(seg_path)
        seg = seg.get_fdata().astype(np.uint8)[:, :, index]

        data = dict()
        data['img'] = torch.from_numpy(img)
        data['seg'] = torch.from_numpy(seg)

        return data['img'], data['seg']

    def _find_entry_by_index(self, index: int) -> tuple[str, IndexInfo, int]:
        offset = 0
        for patient, info in self.index_info.items():
            if offset <= index < offset+info.n_slice:
                return patient, info, index-offset
            offset += info.n_slice
        raise IndexError


@register_dataset
def acdc(root: str, **kwargs) -> tuple[ACDC, dict]:
    dataset = ACDC(root, **kwargs)
    cfg = dict()
    return dataset, cfg
