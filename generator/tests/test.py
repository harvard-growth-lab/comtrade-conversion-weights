import pandas as pd
import glob
import re
from collections import defaultdict
from pathlib import Path
import logging


class TestData:
    def __init__(self, data_dir, logger=None):
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger(__name__)

    def test_dimensions(self):
        """
        test dimension alignment required for matlab optimization code to run
        """
        matrices_dir = self.data_dir / "matrices"
        files = matrices_dir.glob("*.csv")
        combinations = self.extract_year_group_combinations(files)

        # Track pass/fail counts
        passed = 0
        failed = 0
        errors = 0

        # Results storage
        results = {}

        for start_year, end_year, group in sorted(combinations):
            try:
                # Ensure that all required files exist for this combination
                con_path = (
                    matrices_dir
                    / f"conversion.matrix.start.{start_year}.end.{end_year}.group.{group}.csv"
                )
                source_path = (
                    matrices_dir
                    / f"source.trade.matrix.start.{start_year}.end.{end_year}.group.{group}.csv"
                )
                target_path = (
                    matrices_dir
                    / f"target.trade.matrix.start.{start_year}.end.{end_year}.group.{group}.csv"
                )

                try:
                    con = pd.read_csv(con_path)
                    source = pd.read_csv(source_path)
                    target = pd.read_csv(target_path)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"File not found: {con_path}, {source_path}, {target_path}"
                    )

                con_cols_match = con.shape[1] == target.shape[1]
                con_rows_match = con.shape[0] + 1 == source.shape[1]

                status = "PASS" if (con_cols_match and con_rows_match) else "FAIL"
                if status == "FAIL":
                    missing_target_codes = []
                    missing_source_codes = []
                    if not con_cols_match:
                        missing_target_codes = list(
                            set(con.columns[1:]) - set(target.columns[1:])
                        )
                    elif not con_rows_match:
                        missing_source_codes = list(
                            set(con.columns[1:]) - set(source.columns[1:])
                        )

                    results[group] = {
                        "start_year": start_year,
                        "end_year": end_year,
                        "con_shape": con.shape,
                        "source_shape": source.shape,
                        "target_shape": target.shape,
                        "columns_match": con_cols_match,
                        "rows_match": con_rows_match,
                        "missing_target_codes": missing_target_codes,
                        "missing_source_codes": missing_source_codes,
                    }

                    failed += 1
                else:
                    passed += 1
            except Exception as e:
                errors += 1
                self.logger.error(
                    f"Error processing years {start_year}-{end_year}, group {group}: {e}"
                )

        self.logger.info(f"\nTest Summary:")
        self.logger.info(f"Passed: {passed}, Failed: {failed}, Errors: {errors}")

        if failed > 0:
            self.logger.info("\nFailed Tests:")
            for group, result in results.items():
                self.logger.info(
                    f"Years {result['start_year']}-{result['end_year']}, Group {group}:"
                )
                self.logger.info(
                    f"  con shape: {result['con_shape']}, source shape: {result['source_shape']}, target shape: {result['target_shape']}"
                )
                self.logger.info(
                    f"  columns match: {result['columns_match']}, rows match: {result['rows_match']}"
                )

        return results

    def extract_year_group_combinations(self, filenames):
        combinations = set()

        pattern = r"start\.(\d+)\.end\.(\d+)\.group\.(\d+)\.csv"

        for file in filenames:
            match = re.search(pattern, file.name)
            if match:
                start_year = match.group(1)
                end_year = match.group(2)
                group_num = match.group(3)

                # Add this combination to the set
                combinations.add((start_year, end_year, group_num))

        return combinations

    def validate_weights_sum_to_one(self):
        weight_files = self.data_dir / "output" / "grouped_weights"
        for file in weight_files.iterdir():
            self.logger.info(file.name)
            try:
                df = pd.read_csv(file)
            except:
                continue
            source_col = file.name.split(":")[0][-2:]
            self.logger.info(source_col)
            df = df.groupby(source_col).agg({"weight": "sum"})
            df.weight = df.weight.astype(int)
            weights = df.weight.unique()
            self.logger.info(weights)
            if len(weights) > 1:
                self.logger.info(f"review {file}")
