import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from einops import rearrange

from datasets._factory import register_dataset
from datasets.heart.dataset import HeartDataset, IndexInfo


class MnMs1(HeartDataset):
    def __init__(
        self,
        root: str = './data',
    ) -> None:
        super().__init__()
        self.root = Path(root, 'MnMs1').expanduser()

    def _create_index(self) -> dict[str, IndexInfo]:
        index_info = dict()

        cfgs = Path.read_text(self.raw_folder/'201014_M&Ms_Dataset_Information_-_opendataset.csv')
        for folder in ('Training/Labeled', 'Validation', 'Testing',):
            for patient in Path.iterdir(self.raw_folder/folder):
                info = IndexInfo()

                info.fpath = str(patient.relative_to(self.raw_folder))
                seg_path = patient/f'{patient.name}_sa_gt.nii.gz'
                seg = nib.load(seg_path)
                seg = seg.get_fdata().astype(np.uint8)
                seg = rearrange(seg, 'h w c f -> f c h w')

                cfg = re.findall(f'{patient.name},.*,.*,.*,\\d+,\\d+', cfgs)[0]
                cfg = cfg.split(',')
                frame = int(cfg[-2])
                setattr(info, 'ED', [c for c, slice in enumerate(seg[frame]) if slice.max() > 0])
                frame = int(cfg[-1])
                setattr(info, 'ES', [c for c, slice in enumerate(seg[frame]) if slice.max() > 0])

                index_info[patient.name] = info

        return index_info

    def __len__(self) -> int:
        return sum(info.n_slice for _, info in self.index_info.items())

    def __getitem__(self, index: int) -> ...:
        patient, info, index = self._find_entry_by_index(index)
        folder = self.raw_folder/info.fpath

        cfgs = Path.read_text(self.raw_folder/'201014_M&Ms_Dataset_Information_-_opendataset.csv')
        cfg = re.findall(f'{patient},.*,.*,.*,\\d+,\\d+', cfgs)[0]
        cfg = cfg.split(',')
        frame = int(cfg[-2 if index < len(info.ED) else -1])
        index = index if index < len(info.ED) else index-len(info.ED)

        img_path = folder/f'{patient}_sa.nii.gz'
        img = nib.load(img_path)
        img = img.get_fdata().astype(np.float32)[:, :, index, frame]

        seg_path = folder/f'{patient}_sa_gt.nii.gz'
        seg = nib.load(seg_path)
        seg = seg.get_fdata().astype(np.uint8)[:, :, index, frame]

        data = dict()
        data['img'] = torch.from_numpy(img)
        data['seg'] = torch.from_numpy(seg)

        return data['img'], data['seg']


@register_dataset
def mnms1(root: str, **kwargs) -> tuple[MnMs1, dict]:
    dataset = MnMs1(root, **kwargs)
    cfg = dict()
    return dataset, cfg
