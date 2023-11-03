from collections.abc import Callable
from typing import List

import hydra

from omegaconf.dictconfig import DictConfig

import src.augmentations.spectrogram_augmentations
import src.augmentations.wave_augmentations
from src.augmentations.sequential import SequentialAugmentation


def from_configs(config: DictConfig):
    wave_augs = []
    if "augmentations" in config and "wave" in config["augmentations"]:
        for aug_name, aug in config["augmentations"]["wave"].items():
            wave_augs.append(
                hydra.utils.instantiate(aug)
            )

    spec_augs = []
    if "augmentations" in config and "spectrogram" in config["augmentations"]:
        for aug_name, aug in config["augmentations"]["spectrogram"].items():
            spec_augs.append(
                hydra.utils.instantiate(aug)
            )
    return _to_function(wave_augs), _to_function(spec_augs)


def _to_function(augs_list: List[Callable]):
    if len(augs_list) == 0:
        return None
    elif len(augs_list) == 1:
        return augs_list[0]
    else:
        return SequentialAugmentation(augs_list)
