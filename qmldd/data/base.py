from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    """Container for a loaded biomedical dataset.

    y is binary: 1 = disease/positive class, 0 = healthy/negative class.
    This convention is what sensitivity/specificity in qmldd.evaluation assumes.
    """

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    positive_label: str
    negative_label: str


class DataLoader(ABC):
    """Interface every dataset plugin must implement.

    Adding a new disease/dataset to the platform means implementing this
    class and registering it in qmldd.data.DATA_LOADERS - nothing else in
    the pipeline needs to change.
    """

    name: str

    @abstractmethod
    def load(self) -> Dataset: ...
