"""Dataset registry for the dataset-agnostic strict-UDA trainer."""

from dataclasses import dataclass

from datasets.HUST import HUST, speed_dict
from datasets.SJTU import SJTU, condition_dict
from datasets.XJTUSuprgear import XJTUSuprGear, speed_dict as xjtu_speed_dict


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_class: type
    default_data_dir: str
    domain_ids: tuple

    @property
    def num_classes(self):
        return int(self.dataset_class.num_classes)


_DATASETS = {
    "HUST": DatasetSpec(
        "HUST", HUST, "/sda1/FD_GNN/data/HUST", tuple(sorted(speed_dict))
    ),
    "SJTU": DatasetSpec(
        "SJTU", SJTU, "/sda1/FD_GNN/data/SJTU", tuple(sorted(condition_dict))
    ),
    "XJTUSUPRGEAR": DatasetSpec(
        "XJTUSuprGear",
        XJTUSuprGear,
        "/home/baxter/XJTUSpurgear",
        tuple(sorted(xjtu_speed_dict)),
    ),
}


def dataset_names():
    return tuple(sorted(spec.name for spec in _DATASETS.values()))


def get_dataset_spec(name):
    key = str(name).upper()
    if key not in _DATASETS:
        raise ValueError(
            f"unsupported dataset {name!r}; choose from {', '.join(dataset_names())}"
        )
    return _DATASETS[key]
