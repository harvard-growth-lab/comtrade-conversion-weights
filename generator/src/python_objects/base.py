"the parent object for the weight generator"

import os
import sys
from pathlib import Path
import pandas as pd
import typing
import glob
import logging
import shutil
from datetime import datetime
from user_config import LOG_LEVEL
from user_config import RAW_DOWNLOADED_COMTRADE_DATA_PATH

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.set_option("max_colwidth", 400)


class Base(object):


    CLASSIFICATION_FAMILIES = {
        "SITC1": "comtrade", "SITC2": "comtrade", "SITC3": "comtrade",
        "HS1992": "comtrade", "HS1996": "comtrade", "HS2002": "comtrade",
        "HS2007": "comtrade", "HS2012": "comtrade", "HS2017": "comtrade",
        "HS2022": "comtrade",
        "NAICS1997": "naics", "NAICS2002": "naics", "NAICS2007": "naics",
        "NAICS2012": "naics", "NAICS2017": "naics", "NAICS2022": "naics",
    }
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
        "NAICS1997": 1997,
        "NAICS2002": 2002,
        "NAICS2007": 2007,
        "NAICS2012": 2012,
        "NAICS2017": 2017,
        "NAICS2022": 2022,
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
        "NAICS1997": "NAICS1997",
        "NAICS2002": "NAICS2002",
        "NAICS2007": "NAICS2007",
        "NAICS2012": "NAICS2012",
        "NAICS2017": "NAICS2017",
        "NAICS2022": "NAICS2022",
    }

    DETAIL_PRODUCT_CODE_LENGTH = {
        # Comtrade - SITC is 4, HS is 6
        "SITC1": 4, "SITC2": 4, "SITC3": 4,
        "HS1992": 6, "HS1996": 6, "HS2002": 6,
        "HS2007": 6, "HS2012": 6, "HS2017": 6, "HS2022": 6,
        # NAICS - always 6
        "NAICS1997": 6, "NAICS2002": 6, "NAICS2007": 6,
        "NAICS2012": 6, "NAICS2017": 6, "NAICS2022": 6,
    }

    HAS_LEADING_ZEROS = {
        # Comtrade - yes
        "SITC1": True, "SITC2": True, "SITC3": True,
        "HS1992": True, "HS1996": True, "HS2002": True,
        "HS2007": True, "HS2012": True, "HS2017": True, "HS2022": True,
        # NAICS - no
        "NAICS1997": False, "NAICS2002": False, "NAICS2007": False,
        "NAICS2012": False, "NAICS2017": False, "NAICS2022": False,
    }

    # maintained for backwards compatibility
    NAICS_DETAIL_PRODUCT_CODE_LENGTH = 6
    SITC_DETAIL_PRODUCT_CODE_LENGTH = 4
    HS_DETAIL_PRODUCT_CODE_LENGTH = 6
    SITC_YEAR_CUTOFF = 1988


    def __init__(self, conversion_weights_pairs, data_source):
        self.conversion_weights_pairs = conversion_weights_pairs
        self.data_source = data_source if data_source is not None else "comtrade"
        self.root_dir = Path(__file__).parent.parent.parent.absolute()
        sys.path.insert(0, str(self.root_dir))

        # PATHS
        self.data_path = self.root_dir / "data"
        self.static_data_path = self.data_path / "static"
        self.output_path = self.data_path / "output"
        self.downloaded_comtrade_data_path = Path(RAW_DOWNLOADED_COMTRADE_DATA_PATH)
        self.correlation_groups_path = self.data_path / "correlation_groups"
        self.conversion_weights_path = self.data_path / "conversion_weights"
        for path in [
            self.data_path,
            self.static_data_path,
            self.output_path,
            self.downloaded_comtrade_data_path,
            self.correlation_groups_path,
            self.conversion_weights_path,
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


    def get_source_and_target_years(self) -> None:
        """
        Sets the source and target years for the conversion weight pair.
        """

        if self.conversion_weight_pair["direction"] == "backward":
            # H1 => H0, source 1995 & target 1996
            self.source_year = self.RELEASE_YEARS[self.source_class]
            self.target_year = self.source_year - 1
        else:
            # forward direction example: H0 => H1, source 1995 & target 1996
            self.target_year = self.RELEASE_YEARS[self.target_class]
            self.source_year = self.target_year - 1

