import pandas as pd
import glob
import re
from collections import defaultdict
from pathlib import Path

def test_dimensions():
    """
    test dimension alignment required for matlab optimization code to run
    """
    data_dir = "/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/matrices"
    files = glob.glob(f"{data_dir}/*.csv")
    combinations = extract_year_group_combinations(files)
    
    # Track pass/fail counts
    passed = 0
    failed = 0
    errors = 0
    
    # Results storage
    results = {}
    
    for start_year, end_year, group in sorted(combinations):
        try:
            # Ensure that all required files exist for this combination
            con_path = f"{data_dir}/conversion.matrix.start.{start_year}.end.{end_year}.group.{group}.csv"
            source_path = f"{data_dir}/source.trade.matrix.start.{start_year}.end.{end_year}.group.{group}.csv"
            target_path = f"{data_dir}/target.trade.matrix.start.{start_year}.end.{end_year}.group.{group}.csv"
            
            # Check if all files exist before proceeding
            for path in [con_path, source_path, target_path]:
                if not glob.glob(path):
                    raise FileNotFoundError(f"File not found: {path}")
            
            # Load dataframes
            con = pd.read_csv(con_path)
            source = pd.read_csv(source_path)
            target = pd.read_csv(target_path)
            
            # Test conditions
            con_cols_match = con.shape[1] == target.shape[1]
            con_rows_match = con.shape[0] + 1 == source.shape[1]

            status = 'PASS' if (con_cols_match and con_rows_match) else 'FAIL'
            if status == 'FAIL':
                missing_target_codes = []
                missing_source_codes = []
                if not con_cols_match:
                    missing_target_codes = list(set(con.columns[1:]) - set(target.columns[1:]))
                elif not con_rows_match:
                    missing_source_codes = list(set(con.columns[1:]) - set(source.columns[1:]))
            
                # Store results
                results[group] = {
                    'start_year': start_year,
                    'end_year': end_year,
                    'con_shape': con.shape,
                    'source_shape': source.shape,
                    'target_shape': target.shape,
                    'columns_match': con_cols_match,
                    'rows_match': con_rows_match,
                    'missing_target_codes': missing_target_codes,
                    'missing_source_codes': missing_source_codes
                }
            
                failed += 1
            else:
                passed += 1
        except Exception as e:
            errors += 1
            print(f"Error processing years {start_year}-{end_year}, group {group}: {e}")
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Passed: {passed}, Failed: {failed}, Errors: {errors}")
    
    # Print detailed failures
    if failed > 0:
        print("\nFailed Tests:")
        for group, result in results.items():
            print(f"Years {result['start_year']}-{result['end_year']}, Group {group}:")
            print(f"  con shape: {result['con_shape']}, source shape: {result['source_shape']}, target shape: {result['target_shape']}")
            print(f"  columns match: {result['columns_match']}, rows match: {result['rows_match']}")
    
    return results

def extract_year_group_combinations(filenames):
    # Dictionary to store all existing combinations of years and groups
    combinations = set()
    
    # Regex pattern to extract years and group numbers
    pattern = r'start\.(\d+)\.end\.(\d+)\.group\.(\d+)\.csv'
    
    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            start_year = match.group(1)
            end_year = match.group(2)
            group_num = match.group(3)
            
            # Add this combination to the set
            combinations.add((start_year, end_year, group_num))
    
    return combinations


def validate_weights_sum_to_one():
    weight_files = Path("/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/output/grouped_weights")
    for file in weight_files.iterdir():
        print(file.name)
        try:
            df = pd.read_csv(file)
        except:
            continue
        source_col = file.name.split(":")[0][-2:]
        print(source_col)
        df = df.groupby(source_col).agg({'weight':"sum"})
        df.weight = df.weight.astype(int) 
        weights = df.weight.unique()
        print(weights)
        if len(weights) > 1:
            print(f"review {file}")



if __name__ == "__main__":
    # Run the test
    results = test_dimensions()