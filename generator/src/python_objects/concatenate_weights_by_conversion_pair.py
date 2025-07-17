import glob
import re
import pandas as pd
from src.utils.util import clean_groups
from src.python_objects.base import Base
import os
from pathlib import Path


class ConcatenateWeights(Base):
    """
    The optimization code outputs files for each group within a conversion
    weight pair.

    concatenates output from the optimization matlab code
    and adds back in the 1:1 and N:1 mapped products

    The output is saved in the output/optimized_conversion_weights folder.
    """

    GROUP_ID_PATTERN = r"group\.(\d+)\.csv$"
    CONVERSION_WEIGHT_FILENAME_PATTERN = (
        "conversion.weights.start.{start}.end.{end}.group.*.csv"
    )
    CONVERSION_MATRIX_FILENAME_PATTERN = (
        "conversion.matrix.start.{start}.end.{end}.group.{group_id}.csv"
    )
    WEIGHT_FOR_N_TO_1_MAPPED_PRODUCTS = 1
    NO_WEIGHT = 0

    def __init__(self, conversion_weight_pair):
        super().__init__(conversion_weight_pair)
        self.source_class = conversion_weight_pair["source_class"]
        self.start_year = conversion_weight_pair["source_year"]
        self.target_class = conversion_weight_pair["target_class"]
        self.end_year = conversion_weight_pair["target_year"]

        self.matrices_dir = self.data_path / "matrices"
        self.groups_dir = self.data_path / "correlation_groups"
        self.final_optimized_weights_dir = (
            self.output_path / "optimized_conversion_weights"
        )
        for path in [
            self.matrices_dir,
            self.groups_dir,
            self.final_optimized_weights_dir,
        ]:
            self.setup_paths(path)
            
        self.target_class_code = self.classification_translation_dict[self.target_class]
        self.source_class_code = self.classification_translation_dict[self.source_class]


    def run(self):
        # for conversion_pair in self.conversion_years:
        optimized_conversion_weights = pd.DataFrame()
        results = self.get_conversion_weight_optimization_results()

        if self.source_class.startswith("H"):
            detailed_product_level = self.HS_DETAIL_PRODUCT_CODE_LENGTH
        else:
            detailed_product_level = self.SITC_DETAIL_PRODUCT_CODE_LENGTH

        for file in results:
            mapped_conversion_weight_df = self.format_conversion_weight_file(
                file, detailed_product_level
            )
            optimized_conversion_weights = pd.concat(
                [optimized_conversion_weights, mapped_conversion_weight_df]
            )

        non_grouped_products = self.add_non_grouped_products(
            optimized_conversion_weights
        )
        optimized_conversion_weights = pd.concat(
            [optimized_conversion_weights, non_grouped_products]
        )
        self.save_optimized_conversion_weights(optimized_conversion_weights)

    def get_conversion_weight_optimization_results(self) -> list[Path]:
        weights_dir = self.data_path / "conversion_weights"
        results = weights_dir.glob(
            self.CONVERSION_WEIGHT_FILENAME_PATTERN.format(
                start=self.start_year, end=self.end_year
            )
        )
        if not results:
            raise ValueError("Missing conversion weights data, need to run step 4")
        return results

    def format_conversion_weight_file(
        self, file: Path, detailed_product_level: int
    ) -> pd.DataFrame:
        """
        Aligns the indices and columns for the conversion results dataframe
        for a single group with the same group_ids input conversion matrix

        Dataframe is then melted to columns of source product code,
        target product code, and weight
        """
        match = re.search(self.GROUP_ID_PATTERN, str(file))
        if not match:
            return pd.DataFrame()
        group_id = match.group(1)

        # Load weights and conversion matrix
        conversion_group = pd.read_csv(
            self.matrices_dir
            / self.CONVERSION_MATRIX_FILENAME_PATTERN.format(
                start=self.start_year, end=self.end_year, group_id=group_id
            ),
            dtype={"code.source": str},
        )
        weights = pd.read_csv(file, header=None)

        # Standardize source product codes
        conversion_group["code.source"] = conversion_group["code.source"].apply(
            lambda x: (
                x.zfill(detailed_product_level)
                if len(x) < detailed_product_level and x != "TOTAL"
                else x
            )
        )

        conversion_group = conversion_group.set_index("code.source")
        weight_df = pd.DataFrame(
            weights.values,
            index=conversion_group.index,
            columns=conversion_group.columns,
        )

        mapped_conversion_weight_df = (
            weight_df.reset_index()
            .melt(
                id_vars="code.source",
                var_name="code.target",
                value_name="weight",
            )
            .astype({"code.source": str, "code.target": str, "weight": float})
        )
        mapped_conversion_weight_df["group_id"] = group_id
        return mapped_conversion_weight_df

    def add_non_grouped_products(
        self, optimized_conversion_weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds back products that did not require optimization and thus never assigned a group id
        """
        groups = pd.read_csv(
            self.groups_dir / f"from_{self.source_class}_to_{self.target_class}.csv",
            dtype={self.source_class: str, self.target_class: str},
        )
        # add back products that did not require optimization and thus never assigned a group id
        non_grouped_products = groups[groups["group.id"].isna()]
        non_grouped_products = clean_groups(
            non_grouped_products, self.source_class, self.target_class
        )

        non_grouped_products = non_grouped_products[
            ["group.id", "code.source", "code.target"]
        ]
        non_grouped_products["weight"] = self.WEIGHT_FOR_N_TO_1_MAPPED_PRODUCTS
        non_grouped_products = non_grouped_products.rename(
            columns={"group.id": "group_id"}
        )
        return non_grouped_products

    def save_optimized_conversion_weights(
        self, optimized_conversion_weights: pd.DataFrame
    ) -> None:
        """
        Saves the optimized conversion weights to the output/optimized_conversion_weights folder.
        """
        optimized_conversion_weights[["code.target", "code.source"]] = (
            optimized_conversion_weights[["code.target", "code.source"]].astype(str)
        )
        optimized_conversion_weights = optimized_conversion_weights.rename(
            columns={"code.target": self.target_class, "code.source": self.source_class}
        )

        self.logger.info(f"saving {self.source_class}_{self.target_class}.csv")
        optimized_conversion_weights = optimized_conversion_weights[
            optimized_conversion_weights.weight != self.NO_WEIGHT
        ]
        optimized_conversion_weights.to_csv(
            self.final_optimized_weights_dir
            / f"conversion_weights_{self.source_class_code}_to_{self.target_class_code}.csv",
            index=False,
        )
