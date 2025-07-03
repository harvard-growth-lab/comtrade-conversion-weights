from src.python_objects.build_input_matrices import MatrixBuilder
from src.python_objects.prep_for_pipeline import MatlabProgramRunner, GroupWeights
from src.python_objects.combine_concordances import CombineConcordances
import os
import argparse
from tests.test import test_dimensions
from src.utils import util
from pathlib import Path
import subprocess

# parse to set which sections to run
parser = argparse.ArgumentParser()
parser.add_argument(
    "--sections",
    type=str,
    default="all",
    help="""Specify which sections to run (comma-separated numbers or 'all'). Available sections:
        1: Combine concordances - Concatenates concordance tables into main file
        2: "NOT IMPLEMENTED": Run R script to create product groups (must be run separately)
        3: Build matrices - Constructs source classification, target classification, and concordance matrices
        4: Run optimization - Validates matrices and runs MATLAB optimization
        5: Group weights - Groups weights by start and end year pairs
        Example: --sections 1,3,5 to run only sections 1, 3, and 5""",
)
args = parser.parse_args()


CONVERSION_PAIRS = [
    # Backward HS conversions
    {
        "direction": "backward",
        "from_class": "H1",
        "to_class": "H0",
        "from_year": "1996",
        "to_year": "1995",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "H2",
        "to_class": "H1",
        "from_year": "2002",
        "to_year": "2001",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "H3",
        "to_class": "H2",
        "from_year": "2007",
        "to_year": "2006",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "H4",
        "to_class": "H3",
        "from_year": "2012",
        "to_year": "2011",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "H5",
        "to_class": "H4",
        "from_year": "2017",
        "to_year": "2016",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "H6",
        "to_class": "H5",
        "from_year": "2022",
        "to_year": "2021",
        "enabled": True,
    },
    # SITC conversions
    {
        "direction": "backward",
        "from_class": "H0",
        "to_class": "S3",
        "from_year": "1992",
        "to_year": "1988",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "S3",
        "to_class": "S2",
        "from_year": "1988",
        "to_year": "1987",
        "enabled": False,
    },
    {
        "direction": "backward",
        "from_class": "S2",
        "to_class": "S1",
        "from_year": "1987",
        "to_year": "1976",
        "enabled": False,
    },
    # Forward HS conversions
    {
        "direction": "forward",
        "from_class": "H0",
        "to_class": "H1",
        "from_year": "1995",
        "to_year": "1996",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "H1",
        "to_class": "H2",
        "from_year": "2001",
        "to_year": "2002",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "H2",
        "to_class": "H3",
        "from_year": "2006",
        "to_year": "2007",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "H3",
        "to_class": "H4",
        "from_year": "2011",
        "to_year": "2012",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "H4",
        "to_class": "H5",
        "from_year": "2016",
        "to_year": "2017",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "H5",
        "to_class": "H6",
        "from_year": "2021",
        "to_year": "2022",
        "enabled": True,
    },
    # Forward SITC conversions
    {
        "direction": "forward",
        "from_class": "S3",
        "to_class": "H0",
        "from_year": "1988",
        "to_year": "1992",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "S2",
        "to_class": "S3",
        "from_year": "1987",
        "to_year": "1988",
        "enabled": False,
    },
    {
        "direction": "forward",
        "from_class": "S1",
        "to_class": "S2",
        "from_year": "1975",
        "to_year": "1976",
        "enabled": False,
    },
]


def run(sections=[1, 2, 3, 4, 5]):
    if "all" in sections or "1" in sections:
        print("Combining concordances")
        # combine concordances
        CombineConcordances().concatentate_concordance_to_main()

    if "all" in sections or "2" in sections:
        # result = subprocess.run(['Rscript', "src/R_scripts/create_product_groups.R"], capture_output=True, check=True)
        # print(result.stdout)
        raise NotImplementedError("Run create_product_groups.R from R script")

    # Filter enabled pairs
    weight_tables = [
        (pair["direction"], pair["from_class"], pair["to_class"])
        for pair in CONVERSION_PAIRS
        if pair["enabled"]
    ]

    conversion_years = [
        (pair["from_class"], pair["from_year"], pair["to_class"], pair["to_year"])
        for pair in CONVERSION_PAIRS
        if pair["enabled"]
    ]

    if "all" in sections or "3" in sections:
        # build source classification, target classification, and concordance matrices
        for source_class, source_year, target_class, target_year in conversion_years:
            matrices_dir = Path("/n/hausmann_lab/lab/atlas/bustos_yildirim/" +
                    "weights_generator/generator/data/matrices")
            
            if matrices_dir.exists():
                matrices_clean_up_files = list(matrices_dir.glob(f'conversion.matrix.start.{source_year}.end.{target_year}.group.*.csv'))
                util.cleanup_files_from_dir(matrices_clean_up_files)

        matrix_builder = MatrixBuilder(weight_tables)
        matrix_builder.build()

    if "all" in sections or "4" in sections:
        # confirm complete matrices before running matlab optimization
        failed_tests = test_dimensions()
        if failed_tests:
            raise ValueError(f"Failed tests: {failed_tests}")
        
        # for source_class, source_year, target_class, target_year in conversion_years:
        #     weights_dir = Path("/n/hausmann_lab/lab/atlas/bustos_yildirim/" +
        #             "weights_generator/generator/data/conversion_weights")
        #     if weights_dir.exists():
        #         conversion_clean_up_files = list(weights_dir.glob(f'conversion.weights.start.{source_year}.end.{target_year}.group.*.csv'))
        #         util.cleanup_files_from_dir(conversion_clean_up_files)

        matlab_runner = MatlabProgramRunner(conversion_years)
        matlab_runner.write_matlab_params()
        matlab_runner.run_matlab_optimization()

    if "all" in sections or "5" in sections:
        print("Grouping weights by start and end year pairs")
        grouper = GroupWeights(conversion_years)
        grouper.run()


if __name__ == "__main__":
    sections = [section for section in args.sections.split(",")]
    run(sections=sections)
