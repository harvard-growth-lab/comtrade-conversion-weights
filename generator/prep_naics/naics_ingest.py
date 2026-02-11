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
        # forward conversion
        self.source_vintage = "2007"
        self.target_vintage = "2012"
        self.df = pd.read_stata(
            self.input_dir / f"naics_{self.target_vintage}_{self.source_vintage}.dta"
        )

    def ingest_correlation_tables(self):
        """
        construct data inputs for R code
        """
        df = self.df.copy()
        count_target_per_source = self.df.groupby(f"naics{self.target_vintage}")[
            f"naics{self.source_vintage}"
        ].transform("nunique")
        count_source_per_target = self.df.groupby(f"naics{self.source_vintage}")[
            f"naics{self.target_vintage}"
        ].transform("nunique")
        df["Relationship"] = np.char.add(
            np.char.add(np.where(count_target_per_source == 1, "1", "n"), ":"),
            np.where(count_source_per_target == 1, "1", "n"),
        )
        # forward conversion
        if self.source_vintage < self.target_vintage:
            df = df.rename(
                columns={
                    f"naics{self.source_vintage}": "code.before",
                    f"naics{self.target_vintage}": "code.after",
                }
            )
        else:
            df = df.rename(
                columns={
                    f"naics{self.target_vintage}": "code.before",
                    f"naics{self.source_vintage}": "code.after",
                }
            )
        df["adjustment"] = f"{self.source_vintage} to {self.target_vintage}"
        return df[["code.before", "code.after", "Relationship", "adjustment"]]
