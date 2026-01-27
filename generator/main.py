import sys
import os
from src.python_objects.build_comtrade_input_matrices import MatrixBuilder as ComtradeMatrixBuilder
from src.python_objects.concatenate_weights_by_conversion_pair import ConcatenateWeights
from src.python_objects.run_weight_optimizer import MatlabProgramRunner
from src.python_objects.combine_correlation_tables import CombineCorrelationTables
from src.utils import util
from pathlib import Path
import subprocess
from user_config import get_enabled_conversions
from user_config import (
    COMBINE_CONCORDANCES,
    CREATE_PRODUCT_GROUPS,
    BUILD_INPUT_MATRICES,
    GENERATE_WEIGHTS,
    GROUP_WEIGHTS,
)
from src.python_objects.base import Base
from tests.test import TestData


def run(data_source):
    """
    Runs the main generator code.

    The generator code is a pipeline that:
    1. Combines the correlation tables provided by Comtrade into a single file
    2. Creates product groups based on the correlation tables
    3. Builds input (trade in source classification, trade in
        target classification, and correlation within group) matrices
    4. Generates weights for the input matrices using a matlab optimization program
    5. Groups the weights by start and end year pairs and saves the final weights

    The generator code is run in the following order:
    1. CombineCorrelationTables
    2. CreateProductGroups
    3. BuildInputMatrices
    4. GenerateWeights
    5. GroupWeights
    """

    try:
        conversion_weights_pairs = get_enabled_conversions()
        base_obj = Base(conversion_weights_pairs, data_source)
        logger = base_obj.logger

        if COMBINE_CONCORDANCES:
            # runs all correlation tables through the concatenation process
            logger.info("Combine correlation tables")
            CombineCorrelationTables().concatenate_tables_to_main()

        if CREATE_PRODUCT_GROUPS:
            logger.info("Creating product groups")
            try:
                result = subprocess.run(
                    ["Rscript", "src/R_code/create_product_groups.R"],
                    capture_output=True,
                    check=True,
                    text=True,
                    env=dict(os.environ, R_QUIET="TRUE")
                )
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"R script error: {e}")

        if BUILD_INPUT_MATRICES:
            logger.info("Building input matrices")
            util.cleanup_input_matrices(base_obj)
            if data_source = "comtrade":
                # build source classification, target classification, and correlation matrices
                for conversion_weight_pair in conversion_weights_pairs:
                    matrix_builder = ComtradeMatrixBuilder(conversion_weight_pair)
                    matrix_builder.build()
            if data_source = "naics":
                for conversion_weight_pair in conversion_weights_pairs:
                    matrix_builder = NAICSMatrixBuiler(conversion_weight_pair)
                    matrix_builder.build()



        if GENERATE_WEIGHTS:
            logger.info("Generating weights")
            # confirm complete matrices before running matlab optimization
            test_data = TestData(base_obj.data_path, logger)
            failed_tests = test_data.test_dimensions()
            if failed_tests:
                raise ValueError(f"Failed tests: {failed_tests}")

            util.cleanup_weight_files(base_obj)
            matlab_runner = MatlabProgramRunner(conversion_weights_pairs)
            matlab_runner.write_matlab_params()
            matlab_runner.run_matlab_optimization()

        if GROUP_WEIGHTS:
            logger.info("Grouping weights by start and end year pairs")
            for conversion_weights_pair in conversion_weights_pairs:
                concatenate_obj = ConcatenateWeights(conversion_weights_pair)
                concatenate_obj.run()

    except ValueError as e:
        print(f"Error: {e}")

        if hasattr(e, "__cause__") and e.__cause__:
            logger.debug(f"Root cause: {type(e.__cause__).__name__}: {e.__cause__}")

        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="user_config")
    args = parser.parse_args()
    if args.config == "naics_user_config":
        data_source = "naics"
    else:
        data_source = "comtrade"
    run(data_source)
