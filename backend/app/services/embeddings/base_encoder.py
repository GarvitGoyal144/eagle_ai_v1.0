from abc import ABC, abstractmethod
import numpy as np


class BaseEncoder(ABC):
    """
    Base interface for all vision encoders.
    """

    @abstractmethod
    def encode_image(self, image) -> np.ndarray:
        pass

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        pass