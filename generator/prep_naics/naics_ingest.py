import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NaicsIngest:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.correlation_table_path = output_dir
        self.correlation_table_path.mkdir(exist_ok=True)
        self.source_vintage = "2007"
        self.target_vintage = "2012"
        self.df = pd.read_stata(self.input_dir / f"naics_{self.source_vintage}_{self.target_vintage}.dta")

    def ingest_correlation_tables(self):
        count_target_per_source = self.df.groupby(f'naics{self.target_vintage}')[f'naics{self.source_vintage}'].transform('nunique')
        count_source_per_target = self.df.groupby(f'naics{self.source_vintage}')[f'naics{self.target_vintage}'].transform('nunique')
        self.df['Relationship'] = (
            np.where(count_target_per_source == 1, '1', 'n') + ':' + 
            np.where(count_source_per_target == 1, '1', 'n')
        )
        self.df[[f"NAICS{self.source_vintage}", f"NAICS{self.target_vintage}", "Relationship"]]
