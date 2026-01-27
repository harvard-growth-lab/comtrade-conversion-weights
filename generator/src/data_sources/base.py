from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class DataSource(ABC):
    """
    abstraction for data inputs.
    """
    @abstractmethod
    def get_correlation_file(
        self,
        classification: str, 
        year: int,
    ) -> pd.DataFrame:
        """
        
        """
        pass

    @abstractmethod
    def get_trade_matrix(
        self, 
        classification: str, 
        year: int
    ) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        - reporter (country ISO3 or state FIPS)
        - product_code
        - value
        """
        pass
    
    @abstractmethod
    def get_reporters(self, classification: str, year: int) -> List[str]:
        """Returns list of available reporters"""
        pass