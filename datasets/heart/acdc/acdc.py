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
        cfg = Path.read_text(folder/'Info.cfg')
        frame = int(re.findall(f'{tag}: (\\d+)', cfg)[0])
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


@register_dataset
def acdc(root: str, **kwargs) -> tuple[ACDC, dict]:
    dataset = ACDC(root, **kwargs)
    cfg = dict()
    return dataset, cfg
