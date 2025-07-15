from src.python_objects.build_input_matrices import MatrixBuilder
from src.python_objects.concatenate_weights_by_conversion_pair import ConcatenateWeights
from src.python_objects.run_weight_optimizer import MatlabProgramRunner
from src.python_objects.combine_correlation_tables import CombineCorrelationTables
from src.utils import util
from pathlib import Path
import subprocess
from src.config.user_config import get_enabled_conversions
from src.config.user_config import (
    COMBINE_CONCORDANCES,
    CREATE_PRODUCT_GROUPS,
    BUILD_INPUT_MATRICES,
    GENERATE_WEIGHTS,
    GROUP_WEIGHTS,
)
from src.python_objects.base import Base
from tests.test import TestData


def run():
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

    conversion_weights_pairs = get_enabled_conversions()
    base_obj = Base(conversion_weights_pairs)
    logger = base_obj.logger

    if COMBINE_CONCORDANCES:
        # runs all correlation tables through the concatenation process
        logger.info("Combining concordances")
        CombineCorrelationTables().concatenate_tables_to_main()

    if CREATE_PRODUCT_GROUPS:
        logger.info("Creating product groups")
        try:
            result = subprocess.run(
                ["Rscript", "src/R_code/create_product_groups.R"],
                capture_output=True,
                check=True,
                text=True,
            )
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error("STDOUT:", e.stdout)
            logger.error("STDERR:", e.stderr)
            logger.error("Return code:", e.returncode)

    if BUILD_INPUT_MATRICES:
        # build source classification, target classification, and correlation matrices
        logger.info("Building input matrices")
        util.cleanup_input_matrices(base_obj)
        for conversion_weight_pair in conversion_weights_pairs:
            matrix_builder = MatrixBuilder(conversion_weight_pair)
            matrix_builder.build()
        # matrix_builder = MatrixBuilder(conversion_weights_pairs)
        # matrix_builder.build()

    if GENERATE_WEIGHTS:
        logger.info("Generating weights")
        # confirm complete matrices before running matlab optimization
        test_data = TestData(base_obj.data_path, logger)
        failed_tests = test_data.test_dimensions()
        if failed_tests:
            raise ValueError(f"Failed tests: {failed_tests}")

        util.cleanup_weight_files(base_obj)
        matlab_runner = MatlabProgramRunner()
        matlab_runner.write_matlab_params()
        matlab_runner.run_matlab_optimization()

    if GROUP_WEIGHTS:
        logger.info("Grouping weights by start and end year pairs")
        for conversion_weights_pair in conversion_weights_pairs:
            concatenate_obj = ConcatenateWeights(conversion_weights_pair)
            concatenate_obj.run()


if __name__ == "__main__":
    run()
