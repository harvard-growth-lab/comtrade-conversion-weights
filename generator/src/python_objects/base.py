"the parent object for the weight generator"

import os
import sys
from pathlib import Path
import pandas as pd
import typing
import glob
import pyarrow.parquet as pq
import logging
import shutil
from datetime import datetime
from src.config.user_config import LOG_LEVEL
from src.config.user_config import RAW_DOWNLOADED_COMTRADE_DATA_PATH

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


class Base(object):
    RELEASE_YEARS = {
        "SITC1": 1962,
        "SITC2": 1976,
        "SITC3": 1988,
        "HS1992": 1992,
        "HS1996": 1996,
        "HS2002": 2002,
        "HS2007": 2007,
        "HS2012": 2012,
        "HS2017": 2017,
        "HS2022": 2022,
    }

    classification_translation_dict = {
        "SITC1": "S1",
        "SITC2": "S2",
        "SITC3": "S3",
        "HS1992": "H0",
        "HS1996": "H1",
        "HS2002": "H2",
        "HS2007": "H3",
        "HS2012": "H4",
        "HS2017": "H5",
        "HS2022": "H6",
    }
    SITC_DETAIL_PRODUCT_CODE_LENGTH = 4
    HS_DETAIL_PRODUCT_CODE_LENGTH = 6
    SITC_YEAR_CUTOFF = 1988

    def __init__(self, conversion_weights_pairs):
        self.conversion_weights_pairs = conversion_weights_pairs
        self.root_dir = Path(__file__).parent.parent.parent.absolute()
        sys.path.insert(0, str(self.root_dir))

        # PATHS
        self.data_path = self.root_dir / "data"
        self.static_data_path = self.data_path / "static"
        self.output_path = self.data_path / "output"
        self.downloaded_comtrade_data_path = Path(RAW_DOWNLOADED_COMTRADE_DATA_PATH)
        for path in [
            self.data_path,
            self.static_data_path,
            self.output_path,
            self.downloaded_comtrade_data_path,
        ]:
            self.setup_paths(path)

        # Set up logging based on config
        logging.basicConfig(level=getattr(logging, LOG_LEVEL))
        logger = logging.getLogger(__name__)
        self.logger = self.setup_logging()

    def setup_logging(self):
        """Configure logging with both console and file output"""

        (self.root_dir / "logs").mkdir(exist_ok=True)

        detailed_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        simple_format = logging.Formatter("%(levelname)s: %(message)s")

        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, LOG_LEVEL))

        root_logger.handlers.clear()

        # Console handler (simple format)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_format)
        root_logger.addHandler(console_handler)

        # File handler (detailed format)
        log_file = (
            self.root_dir
            / "logs"
            / f"weights_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_format)
        root_logger.addHandler(file_handler)
        return logging.getLogger(__name__)

    def setup_paths(self, path):
        path.mkdir(parents=True, exist_ok=True)

    def load_parquet(self, data_folder, table_name: str):
        read_dir = self.path_mapping[data_folder]
        if read_dir.exists():
            df = pd.read_parquet(Path(read_dir / f"{table_name}.parquet"))
        else:
            raise ValueError("{data_folder} is not a valid data folder")
        return df

    def save_parquet(
        self,
        df,
        data_folder,
        table_name: str,
        parent_folder="",
    ):
        save_dir = self.path_mapping[data_folder]
        save_dir.mkdir(exist_ok=True)
        if data_folder == "final":
            save_dir = save_dir / parent_folder
            save_dir.mkdir(exist_ok=True)

        save_path = save_dir / f"{table_name}.parquet"
        df.to_parquet(save_path, index=False)
