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


class MatrixBuilder(Base):
    atlas_classifications = ["HS1992", "HS2012", "SITC1", "SITC2"]
    AVERAGE_RANGE = 3

    def __init__(self, conversion_weights_pair):

        super().__init__(conversion_weights_pair)
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
        self.target_class_code = self.classification_translation_dict[self.target_class]
        self.source_class_code = self.classification_translation_dict[self.source_class]
        self.target_class_path = (
            self.downloaded_comtrade_parquet_path / self.target_class_code
        )
        self.source_class_path = (
            self.downloaded_comtrade_parquet_path / self.source_class_code
        )
        self.aggregated_by_year_not_converted_path = Path(
            self.downloaded_comtrade_data_path
            / "aggregated_by_year_not_converted"
            / "parquet"
        )

        self.correlation_groups_path = self.data_path / "correlation_groups"
        self.setup_paths(self.correlation_groups_path)

    def build(self):
        """
        generates conversion and trade values matrices that is
        ready for matlab code to generate conversion weights
        """
        # for conversion_weight_pair in self.conversion_weights_pairs:

        self.get_source_and_target_years()
        files_target, files_source = self.get_trade_files_by_classification()

        self.logger.info(
            f"{self.target_class}: {self.target_year} to {self.source_class}: {self.source_year}"
        )
        files_target = self.get_files_by_classification_in_year(
            files_target, self.target_class_code
        )
        files_source = self.get_files_by_classification_in_year(
            files_source, self.source_class_code
        )
        comtrade_dict = {
            self.target_year: files_target,
            self.source_year: files_source,
        }

        reporters = self.extract_reporters_with_timely_classification_update(
            comtrade_dict
        )
        self.logger.info(
            f"There are {len(reporters)} reporters who switched timely from {self.target_class} to {self.source_class}"
        )

        groups = self.get_combined_correlation_file()
        # extract products that are not grouped all 1:1 and some N:1 relationships
        grouped_products = self.filter_for_only_grouped_products(groups)

        target_dfs = self.prep_trade_dataframes(
            "target",
            self.target_class,
            self.target_year,
            grouped_products,
            reporters,
        )
        source_dfs = self.prep_trade_dataframes(
            "source",
            self.source_class,
            self.source_year,
            grouped_products,
            reporters,
        )

        target_dfs, source_dfs = self.align_reporter_indices(
            grouped_products, target_dfs, source_dfs
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

    def get_source_and_target_years(self) -> None:
        """
        Sets the source and target years for the conversion weight pair.
        """
        if self.source_class in ["HS1992", "SITC3"] and self.target_class in [
            "HS1992",
            "SITC3",
        ]:

            if self.source_class == "HS1992":
                self.source_year = 1992
                self.target_year = 1988
            else:
                self.source_year = 1988
                self.target_year = 1992

        elif self.conversion_weight_pair["direction"] == "backward":
            # H1 => H0, source 1995 & target 1996
            self.source_year = self.RELEASE_YEARS[self.source_class]
            self.target_year = self.source_year - 1
        else:
            # forward direction example: H0 => H1, source 1995 & target 1996
            self.target_year = self.RELEASE_YEARS[self.target_class]
            self.source_year = self.target_year - 1

    def get_trade_files_by_classification(self) -> tuple[list, list]:
        """
        Gets the trade file paths by classification.

        SITC to HS conversions trade files are averaged over the constant AVERAGE_RANGE years
        this is to ensure all products are captured, a requirement for the optimization code
        """
        if self.source_class in ["HS1992", "SITC3"] and self.target_class in [
            "HS1992",
            "SITC3",
        ]:

            if self.source_class == "HS1992":
                source_year = 1992
                target_year = 1988
            else:
                source_year = 1988
                target_year = 1992

            files_target = []
            files_source = []
            # HS to SITC conversions trade files are averaged over 3 years
            # this is to ensure all products are captured, a requirement for the optimization code
            for year in range(target_year, target_year + self.AVERAGE_RANGE):
                target_class_year_path = self.target_class_path / str(year)
                files_target += target_class_year_path.glob("*.parquet")

            for year in range(source_year - 1, source_year + (self.AVERAGE_RANGE - 1)):
                source_class_year_path = self.source_class_path / str(year)
                files_source += source_class_year_path.glob("*.parquet")

        else:
            target_class_year_path = self.target_class_path / str(self.target_year)
            source_class_year_path = self.source_class_path / str(self.source_year)
            files_target = target_class_year_path.glob("*.parquet")
            files_source = source_class_year_path.glob("*.parquet")
        return files_target, files_source

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
        grouped_products = clean_groups(
            grouped_products, self.source_class, self.target_class
        )
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

    def prep_trade_dataframes(
        self,
        classification_type: str,
        classification: str,
        year: int,
        grouped_products: pd.DataFrame,
        reporters: list,
    ) -> pd.DataFrame:
        """
        prepares source and target dataframe inputs for the optimization code
        returning country by product trade dataframe
        """
        df = self.get_trade_dataframe(classification, year)
        # extract timely country reporters that switched to new classification year of release
        df = self.filter_df_for_reporters(classification, df, reporters)

        return self.country_by_prod_trade(
            df, grouped_products, classification_type, classification
        )

    def get_trade_dataframe(self, classification: str, year: int) -> pd.DataFrame:
        """
        Requires the trade data to be downloaded and aggregated by year
        and not converted to the target classification. Use ComtradeDownloader to download the data.
        """
        if self.source_class in ["HS1992", "SITC3"] and self.target_class in [
            "HS1992",
            "SITC3",
        ]:
            return self.generate_year_avgs(classification, year)
        else:
            class_code = self.classification_translation_dict[classification]
            trade_path = Path(
                self.aggregated_by_year_not_converted_path,
                class_code,
                f"{class_code}_{year}.parquet",
            )
            try:
                df = pd.read_parquet(trade_path)
            except FileNotFoundError:
                self.logger.error(f"File not found: {trade_path}")
                self.logger.error(f"Run ComtradeDownloader to download the data")
            return df

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
            f"{table}.matrix.start.{source_year}.end.{target_year}.group.*.csv"
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
                / f"{table}.matrix.start.{source_year}.end.{target_year}.group.{group_id}.csv"
            )

    def align_reporter_indices(
        self, groups: pd.DataFrame, target_dfs: dict, source_dfs: dict
    ) -> tuple[dict, dict]:
        """
        Aligns the reporter indices for the target and source dataframes.

        This is to ensure that the reporter indices are the same for the target and source dataframes.
        This is a requirement for the optimization code.
        """
        # enforces shared reporter indices
        for group_id in groups["group.id"].unique().tolist():
            tdf = target_dfs[group_id].reset_index()
            sdf = source_dfs[group_id].reset_index()
            target_dfs[group_id] = (
                tdf.merge(sdf[["reporterISO3"]], on="reporterISO3", how="outer")
                .set_index("reporterISO3")
                .sort_index()
            )
            source_dfs[group_id] = (
                sdf.merge(tdf[["reporterISO3"]], on="reporterISO3", how="outer")
                .set_index("reporterISO3")
                .sort_index()
            )

        return target_dfs, source_dfs

    def get_files_by_classification_in_year(
        self, files: list, classification: str
    ) -> list:
        """
        Extracts lists of files by classification in a given year.
        """
        files_classification_year = []
        files = [file.name for file in files]
        for file in files:
            f = ComtradeFile(file)
            if f.classification == classification:
                files_classification_year.append(file)

        if not files_classification_year:
            self.logger.error(
                "Check file path in user_config.py, must download data using ComtradeDownloader"
            )
            raise ValueError(f"No files found for {classification}... exiting program")
        return files_classification_year

    def extract_reporters_with_timely_classification_update(
        self, comtrade_dict: dict
    ) -> list:
        """
        When trade classification systems change (e.g., from HS1992 to HS2012),
        not all countries adopt the new system immediately. Some countries are
        "timely" and switch right away, while others lag behind.

        Country reporters must be identical between target and source input
        dataframes, therefore we identify the reporters meet the below criteria:

            - Used the old system in the year before the change
            - Switched to the new system in the year of the change

        """
        reporters_dict = {}
        for _, files in comtrade_dict.items():
            for file in files:
                f = ComtradeFile(file)
                reporters_dict[str(f.reporter_code)] = reporters_dict.get(
                    str(f.reporter_code), []
                ) + [file]
        return [
            key
            for key, value in reporters_dict.items()
            if isinstance(value, list) and len(value) >= 2
        ]

    def generate_year_avgs(self, classification: str, start_year: int) -> pd.DataFrame:
        """
        Generates year averages for the trade data.

        This is to ensure that the trade data is complete for the year.

        For SITC to HS conversions,the trade data is averaged because target
        and source years were not comprehensively reporting all products
        """
        df = pd.DataFrame(
            columns=[
                "reporterISO3",
                "partnerISO3",
                "cmdCode",
                "qty",
                "CIFValue",
                "FOBValue",
                "primaryValue",
            ]
        )

        detailed_product_level = get_detailed_product_level(classification)

        self.logger.debug(f"adding starting year {start_year} into the avg")
        classification_code = self.classification_translation_dict[classification]
        df_path = (
            self.aggregated_by_year_not_converted_path
            / f"{classification_code}/{classification_code}_{start_year}.parquet"
        )
        try:
            df = pd.read_parquet(df_path)
        except FileNotFoundError:
            self.logger.error(f"File not found: {df_path}")
            self.logger.error(f"Run ComtradeDownloader to download the data")
            raise

        df = df[(df.flowCode == "M") & (df.digitLevel == detailed_product_level)]
        df = (
            df.groupby(["reporterISO3", "partnerISO3", "cmdCode"])
            .agg(
                {
                    "qty": "sum",
                    "CIFValue": "sum",
                    "FOBValue": "sum",
                    "primaryValue": "sum",
                }
            )
            .reset_index()
        )
        df = df.rename(
            columns={
                "qty": f"qty_{start_year}",
                "CIFValue": f"CIFValue_{start_year}",
                "FOBValue": f"FOBValue_{start_year}",
                "primaryValue": f"primaryValue_{start_year}",
            }
        )

        for year in range(start_year + 1, start_year + self.AVERAGE_RANGE):
            self.logger.debug(f"adding {year} into the avg")
            classification_code = self.classification_translation_dict[classification]
            df_path = (
                self.aggregated_by_year_not_converted_path
                / f"{classification_code}/{classification_code}_{year}.parquet"
            )
            single_year = pd.read_parquet(df_path)
            single_year = single_year[
                (single_year.flowCode == "M")
                & (single_year.digitLevel == detailed_product_level)
            ]
            single_year = (
                single_year.groupby(["reporterISO3", "partnerISO3", "cmdCode"])
                .agg(
                    {
                        "qty": "sum",
                        "CIFValue": "sum",
                        "FOBValue": "sum",
                        "primaryValue": "sum",
                    }
                )
                .reset_index()
            )

            df = single_year.merge(
                df,
                on=["reporterISO3", "partnerISO3", "cmdCode"],
                how="outer",
                suffixes=(f"_{year}", ""),
            )

        for trade_value in ["FOBValue", "primaryValue", "CIFValue", "qty"]:
            matching_columns = [col for col in df.columns if trade_value in col]
            df[f"{trade_value}_avg"] = df[matching_columns].mean(axis=1)
            df = df.drop(columns=matching_columns)
            df = df.rename(columns={f"{trade_value}_avg": trade_value})

        df["flowCode"] = "M"
        df["digitLevel"] = detailed_product_level
        return df

    def filter_df_for_reporters(
        self, classification: str, df: pd.DataFrame, reporters: list
    ) -> pd.DataFrame:
        """
        Filters the dataframe for the list of reporters identified as timely.
        """
        detailed_product_level = get_detailed_product_level(classification)

        # country reporters are more likely to report imports than exports
        df = df[(df.flowCode == "M") & (df.digitLevel == detailed_product_level)]
        reporter = comtradeapicall.getReference("reporter")
        partner = comtradeapicall.getReference("partner")
        reporter = reporter.astype({"reporterCode": str})
        partner = partner.astype({"PartnerCode": str})
        partner.loc[
            partner["PartnerCodeIsoAlpha3"] == "W00", "PartnerCodeIsoAlpha3"
        ] = "WLD"

        df = df.merge(
            reporter[["reporterCode", "reporterCodeIsoAlpha3"]],
            left_on=["reporterISO3"],
            right_on="reporterCodeIsoAlpha3",
            how="left",
        )
        df = df.merge(
            partner[["PartnerCode", "PartnerCodeIsoAlpha3"]],
            left_on=["partnerISO3"],
            right_on="PartnerCodeIsoAlpha3",
            how="left",
        )
        return df[df.reporterCode.isin(reporters)]

    def country_by_prod_trade(self, df, groups, classification_type, prod_class):
        dfs = {}
        for group_id in groups["group.id"].unique():
            group = groups[groups["group.id"] == group_id].copy()
            detailed_product_level = get_detailed_product_level(prod_class)
            if classification_type == "target":
                group.loc[:, "code.target"] = group["code.target"].astype(str)

                group.loc[
                    group["code.target"].str.len() < detailed_product_level,
                    "code.target",
                ] = group["code.target"].str.zfill(detailed_product_level)

                product_codes = group["code.target"].unique().tolist()
                filtered_df = df[df.cmdCode.isin(product_codes)]
                if filtered_df.empty:
                    import pdb

                    pdb.set_trace()
            elif classification_type == "source":
                group.loc[:, "code.source"] = group["code.source"].astype(str)
                group.loc[
                    group["code.source"].str.len() < detailed_product_level,
                    "code.source",
                ] = group["code.source"].str.zfill(detailed_product_level)

                product_codes = group["code.source"].unique().tolist()
                filtered_df = df[df.cmdCode.isin(product_codes)]
                if filtered_df.empty:
                    import pdb

                    pdb.set_trace()

            pivot_df = filtered_df.pivot_table(
                values="primaryValue", index="reporterISO3", columns="cmdCode"
            )
            # pivot_df.fillna(0)
            dfs[group_id] = pivot_df
        return dfs

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

    def extract_classifications(self, filename):
        # Pattern to match classifications: letters followed by optional digits
        pattern = r"([A-Za-z]+\d*)"

        # Find all matches
        matches = re.findall(pattern, filename)

        # Filter out non-classification words like "to", "Conversion", etc.
        classifications = [
            m
            for m in matches
            if m not in ["to", "Conversion", "and", "Correlation", "Tables", "xls"]
        ]
        source, target = classifications[0], classifications[1]
        return source, target

    def determine_relationship(self, concordance_df):
        """
        Since SITC concordances are provided at a combined 4digit and 5digit level.
        We need to roll up to a four digit level. As a result the relationship field
        is recalculated.

        Determine the relationship type between products in a concordance table
        with proper relationship determination based on unique code mappings.

        Returns:
        DataFrame: The input DataFrame with an additional 'Relationship' column
        """
        df = concordance_df.copy()

        before_to_after_dict = {}
        after_to_before_dict = {}

        for _, row in df.iterrows():
            before_code = row["code.before"]
            after_code = row["code.after"]

            if before_code not in before_to_after_dict:
                before_to_after_dict[before_code] = set()
            before_to_after_dict[before_code].add(after_code)

            if after_code not in after_to_before_dict:
                after_to_before_dict[after_code] = set()
            after_to_before_dict[after_code].add(before_code)

        for i, row in df.iterrows():
            before_code = row["code.before"]
            after_code = row["code.after"]

            before_maps_to_multiple = len(before_to_after_dict[before_code]) > 1
            multiple_map_to_after = len(after_to_before_dict[after_code]) > 1

            if not before_maps_to_multiple and not multiple_map_to_after:
                df.at[i, "Relationship"] = "1:1"
            elif before_maps_to_multiple and not multiple_map_to_after:
                df.at[i, "Relationship"] = "n:1"
            elif not before_maps_to_multiple and multiple_map_to_after:
                df.at[i, "Relationship"] = "1:n"
            else:
                df.at[i, "Relationship"] = "n:n"

        return df


class ComtradeFile:
    """Parses and stores Comtrade file metadata."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.name = self.file_path.name
        self._parse_filename()

    def _parse_filename(self) -> None:
        patterns = [
            r"COMTRADE-FINALCLASSIC-CA(?P<reporter>\d{3})(?P<year>\d{4})(?P<classification>\w+)\[(?P<date>[\d-]+)\]",
            r"COMTRADE-FINAL-CA(?P<reporter>\d{3})(?P<year>\d{4})(?P<classification>\w+)\[(?P<date>[\d-]+)\]",
        ]

        for pattern in patterns:
            match = re.match(pattern, self.name)
            if match:
                self.match = match
                self.reporter_code = match.group("reporter")
                self.year = int(match.group("year"))
                self.classification = match.group("classification")
                self.published_date = datetime.strptime(match.group("date"), "%Y-%m-%d")
                return
        raise ValueError(f"File format has not been handled: {self.name}")

    def swap_classification(self, new_classification):
        """
        Swap the classification in the filename and update the object's properties.

        Parameters:
        -----------
        new_classification : str
            The new classification code to use (e.g., 'H0', 'H1', 'S3', etc.)

        Returns:
        --------
        ComtradeFile
            Returns self for method chaining
        """
        old_filename = self.name

        self.classification = new_classification

        if "FINALCLASSIC" in old_filename:
            pattern = r"(COMTRADE-FINALCLASSIC-CA\d{3}\d{4})(\w+)(\[[\d-]+\]\.parquet)"
        else:
            pattern = r"(COMTRADE-FINAL-CA\d{3}\d{4})(\w+)(\[[\d-]+\]\.parquet)"

        # Replace the classification part with the new classification
        self.name = re.sub(pattern, rf"\1{new_classification}\3", old_filename)

        # Update the file_path to match the new name
        self.file_path = self.file_path.parent / self.name

        return self


class ComtradeFiles:
    """Extract file(s) paths based on provided metadata"""

    def __init__(self, files):
        self.files = files

    def get_file_names(self, reporter_code, dates) -> list:
        files = set()
        for f in self.files:
            for date in dates:
                date_str = date.strftime("%Y-%m-%d")
                file = re.search(
                    f".*COMTRADE-FINAL-CA{reporter_code}\\d{{4}}\\w+\\[{date_str}]", f
                )
                try:
                    files.add(file.string)
                except AttributeError as e:
                    pass
        return files
