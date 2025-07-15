import pandas as pd
import glob
import itertools
from pathlib import Path
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix
from src.python_objects.base import Base
from src.utils.util import get_detailed_product_level
from src.config.product_mappings import products_mapped_manually, SPECIAL_CASES


class CombineCorrelationTables(Base):

    MAX_TRUNCATION_ATTEMPTS = 4

    def __init__(self, conversion_weights_pairs={}):
        super().__init__(conversion_weights_pairs={})

    def concatenate_tables_to_main(self):
        """
        RUN TO ADD Comtrade Correlation Tables to A Consolidated/Clean correlation table
        """
        df = pd.DataFrame(
            columns=["code.after", "code.before", "Relationship", "adjustment"]
        )
        dtype_dict = {"code.after": str, "code.before": str}
        dfs = []

        correlation_files = []
        self.correlation_path = self.static_data_path / "comtrade_correlation_tables"
        for file_type in ["*.xls", "*.xlsx"]:
            correlation_files.extend(self.correlation_path.glob(file_type))

        for file in correlation_files:
            df = self.process_correlation_file(file)
            dfs.append(df)

        consolidated_df = pd.concat(dfs)
        consolidated_df = self.clean_data(consolidated_df)
        consolidated_correlation_path = self.output_path / "consolidated_correlation"
        consolidated_correlation_path.mkdir(exist_ok=True)
        consolidated_df.to_csv(
            consolidated_correlation_path
            / "consolidated_comtrade_correlation_tables.csv",
            index=False,
        )

    def process_correlation_file(self, file: Path) -> pd.DataFrame:
        """
        Process a correlation file and return a dataframe.
        """
        source, target = self.extract_classifications(file.name)
        df = pd.read_excel(file, sheet_name=f"Correlation Tables", header=1, dtype=str)
        df = self.name_columns_based_on_direction(df)

        for col in ["code.after", "code.before"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        self.logger.info(f"concatenating {file.name}")
        products_without_wco_mapping = []
        products_not_mapped = pd.DataFrame()
        products_not_mapped.to_csv(
            self.output_path / "not_mapped_products.csv", index=False, mode="a"
        )
        for classification_type, classification in [
            ("source", source),
            ("target", target),
        ]:
            self.logger.debug(f"loading in {classification} products")
            products = pd.read_excel(
                self.static_data_path
                / f"all_products_by_classification/{classification}.xlsx",
                sheet_name="Sheet1",
                dtype=str,
            )
            products_not_mapped = self.find_products_without_wco_mapping(
                classification_type, classification, df, products
            )
            products_without_wco_mapping.append(products_not_mapped)
        products_not_mapped = pd.concat(products_without_wco_mapping)
        products_not_mapped.to_csv(
            self.output_path / "not_mapped_products.csv", index=False, mode="a"
        )

        df["code.before"] = df["code.before"].apply(
            lambda x: x[:-1] if len(x) == 5 else x
        )
        df["code.after"] = df["code.after"].apply(
            lambda x: x[:-1] if len(x) == 5 else x
        )
        df = self.handle_products_not_mapped_by_wco(
            df, products_not_mapped, products_mapped_manually
        )

        df = df.drop_duplicates(subset=["code.after", "code.before", "adjustment"])
        df = self.handle_special_cases(df)
        return self.recalculate_relationship_column(df)

    def extract_classifications(self, filename: str) -> tuple[str, str]:
        """
        Extracts the source and target classifications from Comtrades
        correlation tables in the static data folder.
        """
        # Pattern to match classifications: letters followed by optional digits
        years = re.findall(r"\b\d{4}\b", filename)
        if years:
            source_classification = [
                k for k, v in self.RELEASE_YEARS.items() if v == int(years[0])
            ][0]
            target_classification = [
                k for k, v in self.RELEASE_YEARS.items() if v == int(years[1])
            ][0]
            return source_classification, target_classification
        else:
            pattern = r"([A-Za-z]+\d*)"
            matches = re.findall(pattern, filename)
            classifications = [
                m
                for m in matches
                if m not in ["to", "Conversion", "and", "Correlation", "Tables", "xls"]
            ]
            return classifications[0], classifications[1]

    def recalculate_relationship_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze and classify relationships between product codes

        Analyzes the relationships between product codes in different classification
        systems. It creates a sparse matrix representation of product code mappings
        and classifies relationships into four categories:

        - '1:1': One-to-one mapping between source and target codes
        - 'n:1': Many-to-one mapping (multiple source codes map to one target code)
        - '1:n': One-to-many mapping (one source code maps to multiple target codes)
        - 'n:n': Many-to-many mapping (multiple source codes map to multiple target codes)

        Performed separately for each adjustment period

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing columns:
            - code.after: Target classification codes
            - code.before: Source classification codes
            - adjustment: Period of classification adjustment

        Returns
        -------
        pandas.DataFrame
            DataFrame with added 'Relationship' column indicating the type of mapping
            between source and target codes for each pair.

        """
        result_df = df.copy()
        result_df["Relationship"] = ""

        # Group by adjustment period
        for adjustment_period in df["adjustment"].unique():
            period_mask = df["adjustment"] == adjustment_period
            period_df = df[period_mask]

            unique_after = period_df["code.after"].unique()
            unique_before = period_df["code.before"].unique()

            # index mapping for each code
            after_to_idx = {code: idx for idx, code in enumerate(unique_after)}
            before_to_idx = {code: idx for idx, code in enumerate(unique_before)}

            row_indices = [after_to_idx[code] for code in period_df["code.after"]]
            col_indices = [before_to_idx[code] for code in period_df["code.before"]]
            num_rows, _ = period_df.shape
            data_values = np.ones(num_rows)

            sparse_matrix = csr_matrix(
                (data_values, (row_indices, col_indices)),
                shape=(len(unique_after), len(unique_before)),
                dtype=np.int8,
            )
            row_sums = np.array(sparse_matrix.sum(axis=1)).flatten()
            col_sums = np.array(sparse_matrix.sum(axis=0)).flatten()

            for idx in period_df.index:
                code_after = df.loc[idx, "code.after"]
                code_before = df.loc[idx, "code.before"]

                after_idx = after_to_idx[code_after]
                before_idx = before_to_idx[code_before]

                # gets the number of connections a code has in the target and source
                after_connections = row_sums[after_idx]
                before_connections = col_sums[before_idx]

                if after_connections == 1 and before_connections == 1:
                    relationship_type = "1:1"
                elif after_connections == 1 and before_connections > 1:
                    relationship_type = "n:1"
                elif after_connections > 1 and before_connections == 1:
                    relationship_type = "1:n"
                # remaining are n:n
                else:
                    relationship_type = "n:n"

                result_df.loc[idx, "Relationship"] = relationship_type
        return result_df

    def name_columns_based_on_direction(
        self, correlation: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Name columns based on direction of conversion

        Forward conversion (source classification release year < target classification release year):
            - code.after: target classification
            - code.before: source classification
        Backward conversion (source classification release year > target classification release year):
            - code.after: source classification
            - code.before: target classification
        """
        source_year = correlation.columns[0][-4:]
        target_year = correlation.columns[1][-4:]

        if int(source_year) > int(target_year):
            direction = "backward"
        else:
            direction = "forward"

        if direction == "backward":
            correlation = pd.DataFrame(
                {
                    "code.after": correlation.iloc[:, 0].astype(str),
                    "code.before": correlation.iloc[:, 1].astype(str),
                    "Relationship": correlation.iloc[:, 2],
                    "adjustment": f"{source_year} to {target_year}",
                }
            )
        elif direction == "forward":
            correlation = pd.DataFrame(
                {
                    "code.after": correlation.iloc[:, 1].astype(str),
                    "code.before": correlation.iloc[:, 0].astype(str),
                    "Relationship": correlation.iloc[:, 2],
                    "adjustment": f"{source_year} to {target_year}",
                }
            )
        return correlation

    def get_source_product_level(self, release_year: int) -> int:
        """
        Determine product code length based on year.

        SITC product codes are 4 digits, HS product codes are 6 digits.
        SITC classifications were not released after SITC PRODUCT CUTOFF year
        """
        return (
            self.SITC_DETAIL_PRODUCT_CODE_LENGTH
            if int(release_year) <= self.SITC_YEAR_CUTOFF
            else self.HS_DETAIL_PRODUCT_CODE_LENGTH
        )

    def find_mapped_source_code(
        self, target_code: str, adjustment_period: str, mappings: dict
    ) -> tuple[str, bool]:
        """
        For products that are not given a mapping by the World Customs Organization,
        product code is truncated to move up product level hierarchy.

        Returns:
            tuple: (source_code, was_found) or (None, False) if not found.
            If found, the source code is returned and was_found is True.
            If not found, the source code is None and was_found is False.
        """
        current_code = target_code

        for _ in range(self.MAX_TRUNCATION_ATTEMPTS):
            if current_code in mappings.get(adjustment_period, {}):
                return mappings[adjustment_period][current_code], True
            current_code = current_code[:-1]
        return None, False

    def expand_parent_code(
        self,
        missing_classification_type: str,
        missing_classification: str,
        matched_code: str,
        product_level: int,
    ) -> list[str]:
        """
        Expand a product code to all products codes that are children of the
        the parent product code
        """
        col = "code.after" if missing_classification_type == "source" else "code.before"
        if len(matched_code) < product_level:
            all_products = pd.read_excel(
                self.static_data_path
                / f"all_products_by_classification/{missing_classification}.xlsx",
                sheet_name="Sheet1",
                dtype=str,
            )
            matching_codes = (
                all_products[
                    (all_products.id.str.startswith(matched_code))
                    & (all_products.id.str.len() == product_level)
                ]["id"]
                .unique()
                .tolist()
            )
            matching_codes = [code for code in matching_codes if code.isdigit()]
            return matching_codes if matching_codes else None
        return [matched_code]

    def find_products_without_wco_mapping(
        self,
        classification_type: str,
        classification: str,
        correlation: pd.DataFrame,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Find all products that are not mapped in the correlation tables.

        This is done by comparing the products in the correlation tables to the products in the products
        dataframe downloaded from Comtrade's Product Reference page.
        If a product is not in the correlation tables, it is added to the missing_products list.

        The missing_products list is then returned as a dataframe.
        """
        detailed_product_level = get_detailed_product_level(classification)
        missing_products = []
        adjustment_period = correlation.adjustment.unique().tolist()[0]
        if classification_type == "source":
            col = "code.after"
            release_year = adjustment_period.split(" to ")[0]
        elif classification_type == "target":
            col = "code.before"
            release_year = adjustment_period.split(" to ")[1]
        products = products[products.aggrlevel == f"{detailed_product_level}"]
        # drop products with non digits characters
        products = products[products["id"].str.isdigit()]
        missing_products_df = products[
            ~products["id"]
            .str[:detailed_product_level]
            .isin(correlation[col].str[:detailed_product_level].unique().tolist())
        ]

        new_columns = {
            "adjustment": adjustment_period,
            "missing_classification_type": classification_type,
            "missing_year": release_year,
            "missing_classification": classification,
        }
        missing_products_df = missing_products_df.assign(**new_columns)
        return missing_products_df[
            [
                "adjustment",
                "id",
                "text",
                "missing_classification_type",
                "missing_year",
                "missing_classification",
            ]
        ]

    def handle_products_not_mapped_by_wco(
        self,
        correlation: pd.DataFrame,
        products_not_mapped: pd.DataFrame,
        products_mapped_manually: dict,
    ) -> pd.DataFrame:
        """
        There are products that are no longer supported by the World Customs
        Organization (WCO) and are not mapped in the correlation tables provided
        by Comtrade

        This function maps these products by either:
            - Truncating the product code to move up the product level hierarchy
            and then expand to capture all children of the parent product code
            - Using a manual mapping we've provided

        The function returns a dataframe with the mapped products.
        """
        adjustments = correlation.adjustment.unique()
        # remove adjustment period if not in correlation from products_mapped_manually
        products_mapped_manually = {
            k: v for k, v in products_mapped_manually.items() if k in adjustments
        }

        newly_mapped_product_list = []

        for row in products_not_mapped.itertuples(index=False):
            missing_classification_type = row.missing_classification_type
            missing_classification = row.missing_classification
            adjustment_period = row.adjustment
            code = str(row.id)
            if missing_classification_type == "source":
                missing_year, matching_year = (
                    row.adjustment.split(" to ")[0],
                    row.adjustment.split(" to ")[1],
                )
                matching_classification_type = "target"
                matching_classification = [
                    k
                    for k, v in self.RELEASE_YEARS.items()
                    if v == int(row.adjustment.split(" to ")[1])
                ][0]
            else:
                missing_year, matching_year = (
                    row.adjustment.split(" to ")[1],
                    row.adjustment.split(" to ")[0],
                )
                matching_classification_type = "source"
                matching_classification = [
                    k
                    for k, v in self.RELEASE_YEARS.items()
                    if v == int(row.adjustment.split(" to ")[0])
                ][0]
            # Try to find a manual mapping
            matching_code, found = self.find_mapped_source_code(
                code, adjustment_period, products_mapped_manually
            )

            if not found:
                raise ValueError(f"No mapping found for {code} in {adjustment_period}")
            else:
                # Determine the expected product code length
                product_level = self.get_source_product_level(matching_year)

                matching_code_list = self.expand_parent_code(
                    matching_classification_type,
                    matching_classification,
                    matching_code,
                    product_level,
                )

                if matching_code_list is None:
                    matching_code_list = [code]

            for matching_code in matching_code_list:
                if missing_classification_type == "source":
                    newly_mapped_product_list.append(
                        {
                            "code.after": code,
                            "code.before": matching_code,
                            "adjustment": adjustment_period,
                        }
                    )
                else:
                    newly_mapped_product_list.append(
                        {
                            "code.after": matching_code,
                            "code.before": code,
                            "adjustment": adjustment_period,
                        }
                    )
        if newly_mapped_product_list:
            newly_mapped_products = pd.DataFrame(newly_mapped_product_list)
            newly_mapped_products.to_csv(
                self.output_path / "products_without_wco_mapping_mapped.csv",
                index=False,
                mode="a",
            )
            new_df = pd.concat([correlation, newly_mapped_products], ignore_index=True)
            new_df.drop_duplicates(
                subset=["code.before", "code.after", "adjustment"], inplace=True
            )
            return new_df
        return correlation

    def handle_special_cases(self, df):
        """
        Handle special cases where the product code is not in the correlation table
        """
        for adjustment_period, (code_type, codes) in SPECIAL_CASES.items():
            if adjustment_period in df.adjustment.unique():
                try:
                    df = df[~df[code_type].isin(codes)]
                except:
                    self.logger.error(f"Error dropping {codes} for {adjustment_period}")
        return df

    def clean_data(self, df):
        df = df[~df["code.before"].isin(["I", "II"])]
        df = df[~((df["code.before"] == "nan") & (df["code.after"] == "nan"))]
        return df[~((df["code.before"].isna()) | (df["code.after"].isna()))]
