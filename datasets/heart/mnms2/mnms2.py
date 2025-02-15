import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from einops import rearrange

from datasets._factory import register_dataset
from datasets.heart.dataset import HeartDataset, IndexInfo


class MnMs2(HeartDataset):
    def __init__(
        self,
        root: str = './data',
    ) -> None:
        super().__init__()
        self.root = Path(root, 'MnMs2').expanduser()

    def _create_index(self) -> dict[str, IndexInfo]:
        index_info = dict()

        for folder in ('dataset',):
            for patient in Path.iterdir(self.raw_folder/folder):
                info = IndexInfo()
                info.fpath = str(patient.relative_to(self.raw_folder))

                for tag in ('ED', 'ES',):
                    seg_path = patient/f'{patient.name}_SA_{tag}_gt.nii.gz'
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
        index = index if index < len(info.ED) else index-len(info.ED)

        img_path = folder/f'{patient}_SA_{tag}.nii.gz'
        img = nib.load(img_path)
        img = img.get_fdata().astype(np.float32)[:, :, index]

        seg_path = folder/f'{patient}_SA_{tag}_gt.nii.gz'
        seg = nib.load(seg_path)
        seg = seg.get_fdata().astype(np.uint8)[:, :, index]

        data = dict()
        data['img'] = torch.from_numpy(img)
        data['seg'] = torch.from_numpy(seg)

        return data['img'], data['seg']


@register_dataset
def mnms2(root: str, **kwargs) -> tuple[MnMs2, dict]:
    dataset = MnMs2(root, **kwargs)
    cfg = dict()
    return dataset, cfg
