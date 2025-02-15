import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torchvision.datasets import VisionDataset


@dataclass
class IndexInfo:
    fpath: str = ''
    ED: list[int] = None
    ES: list[int] = None

    @property
    def n_slice(self) -> int:
        return len(self.ED) + len(self.ES)


class HeartDataset(VisionDataset):
    root: Path
    resources: list[tuple[str, str]]
    __IndexInfo: dict[str, IndexInfo]

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.__IndexInfo = dict()

    @property
    def raw_folder(self) -> Path:
        return self.root/'raw'

    @property
    def index_file(self) -> Path:
        return self.raw_folder/'index.json'

    @property
    def index_info(self) -> dict[str, IndexInfo]:
        if not self.__IndexInfo:
            if not self.index_file.exists():
                save_index(self._create_index(), self.index_file)
            self.__IndexInfo = load_index(self.index_file)
        return self.__IndexInfo

    @property
    def processed_folder(self) -> Path:
        return self.root/'processed'

    @property
    def train_file(self) -> Path:
        return self.processed_folder/'train.pt'

    @property
    def valid_file(self) -> Path:
        return self.processed_folder/'valid.pt'

    @property
    def test_file(self) -> Path:
        return self.processed_folder/'test.pt'

    def _create_index(self) -> None:
        raise NotImplementedError

    def _check_legacy_exist(self) -> bool:
        raise NotImplementedError

    def _load_legacy_data(self) -> tuple:
        raise NotImplementedError

    def _load_data(self) -> tuple:
        raise NotImplementedError

    def _find_entry_by_index(self, index: int) -> tuple[str, IndexInfo, int]:
        offset = 0
        for patient, info in self.index_info.items():
            if offset <= index < offset+info.n_slice:
                return patient, info, index-offset
            offset += info.n_slice
        raise IndexError


def save_index(obj: dict[str, IndexInfo], fpath: Path) -> dict[str, IndexInfo]:
    data = {k: asdict(v) for k, v in obj.items()}
    json.dump(data, fpath.open('w'))
    return obj


def load_index(fpath: Path) -> dict[str, IndexInfo]:
    data = json.load(fpath.open('r'))
    obj = {k: IndexInfo(**v) for k, v in data.items()}
    return obj
