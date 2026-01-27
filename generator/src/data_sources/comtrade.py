from generator.src.data_sources.base import DataSource
import pandas as pd
import requests
from generator.src.python_objects.base import Base


class ComtradeDataSource(DataSource):
    """Fetches economic data from Census Bureau API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.census.gov/data"
        self.values = "EMP"
        self.index = "state"

    def ingest_correlation_tables(self):
        ""
        ""
        count_target_per_source = self.df.groupby(f'naics{self.target_vintage}')[f'naics{self.source_vintage}'].transform('nunique')
        count_source_per_target = self.df.groupby(f'naics{self.source_vintage}')[f'naics{self.target_vintage}'].transform('nunique')
        self.df['Relationship'] = (
            np.where(count_target_per_source == 1, '1', 'n') + ':' + 
            np.where(count_source_per_target == 1, '1', 'n')
        )
        self.df[[f"NAICS{self.source_vintage}", f"NAICS{self.target_vintage}", "Relationship"]]

    
    def get_trade_matrix(self, classification: str, year: int) -> pd.DataFrame:
        """
        """
        url = self.base_url + f"/{year}/ecnbasic?get={classification},EMP&for=state:*&key={self.api_key}"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        df = df[df[classification].astype(str).str.len() == Base.DETAIL_PRODUCT_CODE_LENGTH[classification]]
        df[self.values] = df[self.values].astype(int)
        return df.pivot_table((values=self.values,index=self.index,columns=classification))