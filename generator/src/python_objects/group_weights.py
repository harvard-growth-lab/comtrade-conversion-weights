import glob
import re
import pandas as pd
from src.utils.util import clean_groups
from src.python_objects.base import Base
import os


class GroupWeights(Base):
    def __init__(self, conversion_years):
        super().__init__(conversion_years)
        self.conversion_years = conversion_years

    def run(self):
        for conversion_pair in self.conversion_years:
            source_class = conversion_pair["source_class"]
            start_year = conversion_pair["source_year"]
            target_class = conversion_pair["target_class"]
            end_year = conversion_pair["target_year"]

            print(f"source class {source_class} and target class {target_class}")
            combined_result = pd.DataFrame()
            weights_dir = self.data_path / "conversion_weights"
            results = weights_dir.glob(
                f"conversion.weights.start.{start_year}.end.{end_year}.group.*.csv"
            )
            if not results:
                raise ValueError("Missing conversion weights data, need to run step 4")
            for file in results:
                match = re.search(r"group\.(\d+)\.csv$", str(file))
                if not match:
                    continue

                gid = match.group(1)

                # try:
                matrices_dir = self.data_path / "matrices"
                conversion_group = pd.read_csv(
                    matrices_dir
                    / f"conversion.matrix.start.{start_year}.end.{end_year}.group.{gid}.csv",
                    dtype={"code.source": str},
                )

                # Load weights and conversion matrix
                weights = pd.read_csv(file, header=None)

                if source_class.startswith("H"):
                    detailed_product_level = 6
                else:
                    detailed_product_level = 4
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

                # Convert to long format
                weight_long = (
                    weight_df.reset_index()
                    .melt(
                        id_vars="code.source",
                        var_name="code.target",
                        value_name="weight",
                    )
                    .astype({"code.source": str, "code.target": str, "weight": float})
                )

                weight_long["group_id"] = gid
                combined_result = pd.concat([combined_result, weight_long])
                # except:
                #     print("failed")
            groups_dir = self.data_path / "concordance_groups"
            groups = pd.read_csv(
                groups_dir / f"from_{source_class}_to_{target_class}.csv",
                dtype={source_class: str, target_class: str},
            )
            # add back products that did not require optimization and thus never assigned a group id
            non_grouped_products = groups[groups["group.id"].isna()]
            non_grouped_products = clean_groups(
                non_grouped_products, source_class, target_class
            )

            non_grouped_products = non_grouped_products[
                ["group.id", "code.source", "code.target"]
            ]
            non_grouped_products["weight"] = 1
            non_grouped_products = non_grouped_products.rename(
                columns={"group.id": "group_id"}
            )
            combined_result = pd.concat([combined_result, non_grouped_products])

            combined_result[["code.target", "code.source"]] = combined_result[
                ["code.target", "code.source"]
            ].astype(str)
            combined_result = combined_result.rename(
                columns={"code.target": target_class, "code.source": source_class}
            )

            print(f"saving {source_class}:{target_class}.csv")
            combined_result = combined_result[combined_result.weight != 0]
            output_dir = self.data_path / "output" / "grouped_weights"
            os.makedirs(output_dir, exist_ok=True)
            combined_result.to_csv(
                output_dir / f"grouped_{source_class}_{target_class}.csv",
                index=False,
            )
