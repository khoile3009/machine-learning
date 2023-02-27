from enum import Enum
from pathlib import Path


class DatasetType(Enum):
    TXT = "txt"

DATASET_TYPE_MAP = {
    "tiny_shakespeare": DatasetType.TXT
}



class Dataset:
    ROOT_DIR = Path("D:/Data")
    
    @classmethod    
    def get_dataset(cls, dataset_name):
        dataset_type = DATASET_TYPE_MAP.get(dataset_name)
        dataset_path = cls.ROOT_DIR / f"{dataset_name}.txt"
        if dataset_type == DatasetType.TXT:
            with open(dataset_path, "r", encoding="utf-8") as f:
                data = f.read()
        return data
