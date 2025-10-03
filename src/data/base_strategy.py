"""Base strategy interface for data processing using Strategy Pattern."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd


class DataProcessingStrategy(ABC):
    """Abstract base class for data processing strategies."""

    @abstractmethod
    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw data into features and labels.

        Args:
            data: Raw input DataFrame.

        Returns:
            Tuple of (features, labels) as numpy arrays.
        """
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate that data meets processing requirements.

        Args:
            data: Input DataFrame to validate.

        Returns:
            True if data is valid, False otherwise.
        """
        pass
