# -*- coding: utf-8 -*-a

from functools import lru_cache
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class BlackBox:
    """All BlackBoxes are the same."""

    def __init__(self, contents):
        # TODO: use a weak reference for contents
        self._contents = contents

    @property
    def contents(self):
        return self._contents

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(type(self))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env',
                                      env_file_encoding="utf-8",
                                      extra='ignore')
    blur: int
    morph: int
    iterations: int
    green: List[List[int]]
    yellow: List[List[int]]
    kernel_type: int = cv2.MORPH_ELLIPSE
    morph_type: int = cv2.MORPH_CLOSE
    crop_h_min: int = 0
    crop_h_max: int = -1
    crop_w_min: int = 0
    crop_w_max: int = -1
    threshold: int = 0
    noise_threshold_min: int = 0
    noise_threshold_max: int
    merged_threshold_max: int
    gamma: float
    saturation: int
    rows: int
    columns: int
    input_dir: str
    csv_dir: str
    calibration: str


@lru_cache
def _get_settings(blackbox: object, clear: bool = False):
    print("called with args:", blackbox.contents, clear)
    return Settings(_env_file=blackbox.contents)


def get_settings(env_file: str = '.env', clear: bool = False):
    return _get_settings(BlackBox(env_file), clear)


if __name__ == "__main__":
    print(get_settings(env_file="testing.env").iterations)
    print(get_settings())
    print(get_settings().model_dump())
    print(get_settings().blur)
    get_settings().blur += 1
    print(get_settings().blur)
