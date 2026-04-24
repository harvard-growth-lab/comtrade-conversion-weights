import pandas as pd
import numpy as np
import os
import logging
from src.python_objects.base import Base

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NaicsIngest(Base):
    def __init__(self, source, target, data_source="naics"):
        super().__init__(conversion_weights_pairs={}, data_source=data_source)
        self.source_vintage = source
        self.target_vintage = target
        self.df = pd.read_stata(
            self.data_path/ "temp" / f"naics_{self.source_vintage}_{self.target_vintage}.dta"
        )

    def ingest_correlation_tables(self):
        """
        construct data inputs for R code
        """
        df = self.df.copy()

        source_col = f"naics{self.source_vintage}"
        target_col = f"naics{self.target_vintage}"
        targets_per_source = df.groupby(source_col)[target_col].transform("nunique")
        sources_per_target = df.groupby(target_col)[source_col].transform("nunique")

                
        df["Relationship"] = (
            np.where(sources_per_target == 1, "1", "n") + 
            ":" + 
            np.where(targets_per_source == 1, "1", "n")
        )        
        
        # Rename to before/after based on chronology (for R code compatibility)
        older_vintage = min(self.source_vintage, self.target_vintage)
        newer_vintage = max(self.source_vintage, self.target_vintage)

        df = df.rename(
            columns={
                f"naics{older_vintage}": "code.before",
                f"naics{newer_vintage}": "code.after",
            }
        )
        df["adjustment"] = f"{newer_vintage} to {older_vintage}"        
        df =  df[["code.after", "code.before", "Relationship", "adjustment"]]
        consolidated_correlation_path = self.output_path / "consolidated_correlation"
        import pdb; pdb.set_trace()
        df.to_csv(
            consolidated_correlation_path
            / f"consolidated_{self.data_source}_correlation_tables.csv",
            index=False,
        )
