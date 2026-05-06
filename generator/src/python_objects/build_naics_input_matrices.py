# RUN after Groups from R code has been generated in the data/correlation_groups folder
# PREP CONVERSION, TRADE VALUE TABLES AND WEIGHT TABLES

import pandas as pd
import glob
import os
from datetime import datetime
import re
import comtradeapicall
from pathlib import Path
from src.utils.util import clean_groups, get_detailed_product_level
from src.python_objects.base import Base

pd.options.mode.copy_on_write = True
pd.set_option("future.no_silent_downcasting", True)


class NAICSMatrixBuilder(Base):

    atlas_classifications = ["HS1992", "HS2012", "SITC1", "SITC2"]
    AVERAGE_RANGE = 3

    def __init__(self, conversion_weights_pair, data_source="naics"):

        super().__init__(conversion_weights_pair, data_source=data_source)
        self.downloaded_comtrade_data_path = Path(
            self.downloaded_comtrade_data_path / "as_reported"
        )
        self.downloaded_comtrade_parquet_path = (
            self.downloaded_comtrade_data_path / "raw_parquet"
        )
        # set variables
        self.conversion_weight_pair = conversion_weights_pair
        self.source_class = conversion_weights_pair["source_class"]
        self.target_class = conversion_weights_pair["target_class"]
        self.source_year = conversion_weights_pair["source_year"]
        self.target_year = conversion_weights_pair["target_year"]

        self.correlation_groups_path = self.data_path / "correlation_groups"
        self.setup_paths(self.correlation_groups_path)

    def build(self):
        """
        generates conversion and trade values matrices that is
        ready for matlab code to generate conversion weights
        """
        target_file = self.get_reported_data(self.target_year)
        source_file = self.get_reported_data(self.source_year)
        groups = self.get_combined_correlation_file()
        # extract products that are not grouped all 1:1 and some N:1 relationships
        grouped_products = self.filter_for_only_grouped_products(groups)

        target_dfs = self.country_by_prod_trade(
            target_file,
            grouped_products,
            "target",
            prod_class=self.target_year,
        )
        source_dfs = self.country_by_prod_trade(
            source_file,
            grouped_products,
            "source",
            prod_class=self.source_year,
        )

        group_dfs = self.conversion_matrix(grouped_products)

        self.generate_dataframes(
            target_dfs, "target.trade", self.source_year, self.target_year
        )
        self.generate_dataframes(
            source_dfs, "source.trade", self.source_year, self.target_year
        )
        self.generate_dataframes(
            group_dfs, "conversion", self.source_year, self.target_year
        )

    def get_reported_data(self, year):
        """ """
        # temp file
        df = pd.read_stata(
            f"data/temp/naics_{int(self.source_year)}_{int(self.target_year)}.dta"
        )
        df = df[[f"naics{year}", f"emp{year}"]]
        df["reporter"] = "USA"
        return df

    def country_by_prod_trade(self, df, groups, classification_type, prod_class):
        dfs = {}
        for group_id in groups["group.id"].unique():
            group = groups[groups["group.id"] == group_id].copy()
            if classification_type == "target":
                group.loc[:, "code.target"] = group["code.target"].astype(str)

                product_codes = group["code.target"].unique().tolist()
                filtered_df = df[df[f"naics{prod_class}"].isin(product_codes)]
                if filtered_df.empty:
                    raise ValueError("no product codes matched reported data")

            elif classification_type == "source":
                group.loc[:, "code.source"] = group["code.source"].astype(str)

                product_codes = group["code.source"].unique().tolist()
                filtered_df = df[df[f"naics{prod_class}"].isin(product_codes)]
                if filtered_df.empty:
                    raise ValueError("no product codes matched reported data")

            pivot_df = filtered_df.pivot_table(
                values=f"emp{prod_class}",
                index="reporter",
                columns=f"naics{prod_class}",
            )
            dfs[group_id] = pivot_df
        return dfs

    def get_combined_correlation_file(self) -> pd.DataFrame:
        """
        Gets the correlation file for the source and target classes.
        """

        groups = pd.read_csv(
            self.correlation_groups_path
            / f"from_{self.source_class}_to_{self.target_class}.csv"
        )
        if not groups[
            ((groups["code.source"].isna()) | (groups["code.target"].isna()))
        ].empty:
            raise ValueError(
                f"Unexpected NA values need to be handled for {self.source_class} to {self.target_class} \n {groups[((groups['code.source'].isna()) | (groups['code.target'].isna()))]}"
            )
        groups = groups.astype({"code.source": int, "code.target": int}).astype(
            {"code.source": str, "code.target": str}
        )
        return groups

    def filter_for_only_grouped_products(self, groups: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the groups for only the products that are grouped.

        This is to remove 1:1 groupings and single product groupings,
        which are not passed into the optimization code since product mappings
        are known
        """
        grouped_products = groups[groups["group.id"].notna()]
        grouped_products["group.id"] = grouped_products["group.id"].astype(int)
        if (
            1
            in grouped_products.groupby("group.id")
            .agg({"group.id": "count"})["group.id"]
            .unique()
        ):
            raise ValueError(f"grouping of one product is invalid.")
        if "1:1" in grouped_products.Relationship.unique():
            raise ValueError(f"grouping of one to one relationship, is invalid.")
        return grouped_products

    def generate_dataframes(
        self, dfs: dict, table: str, source_year: int, target_year: int
    ):
        """
        Generates the dataframes for the optimization code.

        The dataframes are saved to the data/matrices folder.
        The dataframes are named as follows:
        {table}.matrix.start.{source_year}.end.{target_year}.group.{group_id}.csv

        The dataframes are used to generate the conversion weights.
        """

        matrices_path = self.data_path / "matrices"
        files = matrices_path.glob(
            f"{table}.matrix.{self.data_source}.start.{source_year}.end.{target_year}.group.*.csv"
        )

        # clean out previously generated files
        for file_path in files:
            try:
                # Check if file exists before attempting to delete
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    self.logger.error(f"File not found: {file_path}")
            except Exception as e:
                self.logger.error(f"Error deleting {file_path}: {str(e)}")

        for group_id, df in dfs.items():
            if df.empty:
                # drop nans
                self.logger.debug(f"df is empty for {table} group: {group_id}")
                continue

            df = df.fillna(0)
            os.makedirs(
                self.data_path / "matrices",
                exist_ok=True,
            )
            df.to_csv(
                self.data_path
                / "matrices"
                / f"{table}.matrix.{self.data_source}.start.{source_year}.end.{target_year}.group.{group_id}.csv"
            )

    def conversion_matrix(self, groups):
        # by group
        # rows are the source
        # cols are the target
        dfs = {}
        for group_id in groups["group.id"].unique():
            group = groups[groups["group.id"] == group_id]
            df = group.pivot_table(
                values="group.id", index="code.source", columns="code.target"
            )
            df = df.replace(group_id, True).infer_objects(copy=False)
            df = df.fillna(False)
            dfs[group_id] = df
        return dfs
