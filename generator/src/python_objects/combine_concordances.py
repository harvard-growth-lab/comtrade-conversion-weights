import pandas as pd
import glob
import itertools
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix



class CombineConcordances():

    SITC_DETAIL_PRODUCT_CODE_LENGTH = 4
    HS_DETAIL_PRODUCT_CODE_LENGTH = 6
    SITC_YEAR_CUTOFF = 1988
    MAX_TRUNCATION_ATTEMPTS = 4
    non_concorded_product_file = "data/static/product_missing_concordance.csv"

    def __init__(self):
        self.concatentate_concordance_to_main()
        # self.concordance_files = concordance_files


    def concatentate_concordance_to_main(self):
        """ 
        RUN TO ADD Comtrade Concordance Tables to A Consolidated/Clean concordance table
        """

        df = pd.DataFrame(columns=['code.after','code.before','Relationship','adjustment'])
        dtype_dict = {'code.after': str, 'code.before': str}
        dfs = []
        xls_concordance_files = glob.glob("data/static/comtrade_concordance/*.xls")
        xlsx_concordance_files = glob.glob("data/static/comtrade_concordance/*.xlsx")
        concordance_files = xls_concordance_files + xlsx_concordance_files
        for file in xls_concordance_files + xlsx_concordance_files:
            source, target = self.extract_classifications(file.split('/')[-1])
            df = pd.read_excel(file, sheet_name = f"Correlation Tables", header=1, dtype=str)
            df = self.format_concordance_table(df)

            for col in ['code.after', 'code.before']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            print(f"added {file.split('/')[-1]} to consolidated concordance table")
            non_concorded_df = pd.read_csv(self.non_concorded_product_file, dtype={"id": str})
            df = self.handle_no_concordances(df, non_concorded_df)
            df['code.before'] = df['code.before'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df['code.after'] = df['code.after'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df = df.drop_duplicates(subset=["code.after", "code.before", "adjustment"])
            df = self.add_relationship_column(df)
            dfs.append(df)

        consolidated_df = pd.concat(dfs)

        consolidated_df = consolidated_df[~((consolidated_df['code.before']=='nan') & (consolidated_df['code.after']=='nan'))]
        consolidated_df = consolidated_df[~((consolidated_df['code.before'].isna()) | (consolidated_df['code.after'].isna()))]
        consolidated_df.to_csv(f"data/output/consolidated_concordance/consolidated_comtrade_concordances.csv", index=False)


    def extract_classifications(self,filename):
        # Pattern to match classifications: letters followed by optional digits
        pattern = r'([A-Za-z]+\d*)'
        
        # Find all matches
        matches = re.findall(pattern, filename)
        
        # Filter out non-classification words like "to", "Conversion", etc.
        classifications = [m for m in matches if m not in ["to", "Conversion", "and", "Correlation", "Tables", "xls"]]
        source, target = classifications[0], classifications[1]
        return source, target
    
    def add_relationship_column(self, df):
        """
        Add a relationship column based on sparse matrix analysis 
        grouped by adjustment period
        """
        result_df = df.copy()
        result_df['Relationship'] = ''
        
        # Group by adjustment period
        for adjustment_period in df['adjustment'].unique():
            period_mask = df['adjustment'] == adjustment_period
            period_df = df[period_mask]
            
            unique_after = period_df['code.after'].unique()
            unique_before = period_df['code.before'].unique()
            
            # index mapping for each code
            after_to_idx = {code: idx for idx, code in enumerate(unique_after)}
            before_to_idx = {code: idx for idx, code in enumerate(unique_before)}
            
            row_indices = [after_to_idx[code] for code in period_df['code.after']]
            col_indices = [before_to_idx[code] for code in period_df['code.before']]
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
                code_after = df.loc[idx, 'code.after']
                code_before = df.loc[idx, 'code.before']
                
                after_idx = after_to_idx[code_after]
                before_idx = before_to_idx[code_before]
                
                # gets the number of connections a code has in the target and source
                after_connections = row_sums[after_idx]
                before_connections = col_sums[before_idx]
                
                if after_connections == 1 and before_connections == 1:
                    relationship_type = '1:1'
                elif after_connections == 1 and before_connections > 1:
                    relationship_type = 'n:1'
                elif after_connections > 1 and before_connections == 1:
                    relationship_type = '1:n'
                # remaining are n:n
                else:
                    relationship_type = 'n:n'
                
                result_df.loc[idx, 'Relationship'] = relationship_type
        return result_df
    

    def format_concordance_table(self, concordance):
        source_year = concordance.columns[0][-4:]
        target_year = concordance.columns[1][-4:]

        if int(source_year) > int(target_year):
            direction = "backward"
        else:
            direction = "forward"

        if direction == "backward":
            concordance = pd.DataFrame({
                'code.after': concordance.iloc[:, 0].astype(str),
                'code.before': concordance.iloc[:, 1].astype(str),
                'Relationship': concordance.iloc[:, 2],
                'adjustment': f"{source_year} to {target_year}"
            })
        elif direction == "forward":
            concordance = pd.DataFrame({
                'code.after': concordance.iloc[:, 1].astype(str),
                'code.before': concordance.iloc[:, 0].astype(str),
                'Relationship': concordance.iloc[:, 2],
                'adjustment': f"{source_year} to {target_year}"
            })
        return concordance

    def get_source_product_level(self, year):
        """Determine product code length based on year."""
        return self.SITC_DETAIL_PRODUCT_CODE_LENGTH if int(year) <= self.SITC_YEAR_CUTOFF else self.HS_DETAIL_PRODUCT_CODE_LENGTH


    def find_mapped_source_code(self, target_code, adjustment_period, mappings):
        """
        Try to find a mapping for the target code by progressively truncating it.
        
        Returns:
            tuple: (source_code, was_found) or (None, False) if not found
        """
        current_code = target_code
        
        for _ in range(self.MAX_TRUNCATION_ATTEMPTS):
            if current_code in mappings.get(adjustment_period, {}):
                return mappings[adjustment_period][current_code], True
            current_code = current_code[:-1]
        return None, False


    def expand_partial_code(self, missing_classification_type, missing_classification, code, product_level, adjustment_period, concordance):
        """
        Expand a partial source code to all matching detailed codes.
        """
        col = "code.after" if missing_classification_type == "source" else "code.before"
        if len(code) < product_level:
            all_products = pd.read_excel(f"data/static/all_products_by_classification/{missing_classification}.xlsx", sheet_name="Sheet1", dtype=str)
            matching_codes = all_products[(all_products.id.str.startswith(code))&(all_products.id.str.len()==product_level)]['id'].unique().tolist()
            matching_codes = [code for code in matching_codes if code.isdigit()]
            return matching_codes if matching_codes else None
        return [code]



    def handle_no_concordances(self, concordance, non_concorded_df):
        """
        Handle cases where there is no concordance between the source and target
        """
        non_corded_product_matches = {
            "1976 to 1962": {
                "6514": "6516",
                "9710" : "897",  # rolled up
            },
            "1992 to 1988": {
                "9110": "999999",
                "9310": "999999",
                "334": "2710",
                "2710": "334",
                "380999": "5989",
                "999999": "9999",
                "9999AA": "9999",
                "711890": "9710",
                "710820": "9710",
            },
            "1996 to 1992": {
                "2710": "2710",
                "999999": "999999",
                "9999AA": "999999",
            },
            "2002 to 1996": {
                "2710": "2710",
                "999999": "999999",
                "9999AA": "999999",
            },
            "2007 to 2002": {
                "999999": "999999",
                "9999AA": "999999",
            },
            "2012 to 2007": {
                "291636": "291616",  # Binapacryl (ISO)
                "999999": "999999",
                "9999AA": "999999",
            },
            "2017 to 2012": {
                "999999": "999999",
                "9999AA": "999999",
            },
            "2022 to 2017": {
                "300219": "3002"
            }
        }
        adjustments = concordance.adjustment.unique()
        # remove adjustment period if not in concordance from non_corded_product_matches
        non_corded_product_matches = {k: v for k, v in non_corded_product_matches.items() if k in adjustments}

        concorded_products = []

        for row in non_concorded_df.itertuples(index=False):
            if row.adjustment not in non_corded_product_matches:
                continue
            missing_classification_type = row.missing_classification
            missing_classification = row.missing_product_classification
            adjustment_period = row.adjustment
            code = str(row.id)  # Convert to string for consistency
            if missing_classification_type == "source":
                missing_year = row.adjustment.split(" to ")[0]
            else:
                missing_year = row.adjustment.split(" to ")[1]

            # Try to find a manual mapping
            missing_code, found = self.find_mapped_source_code(
                code, 
                adjustment_period, 
                non_corded_product_matches
            )

            if not found:
                # No mapping found - use the original code
                raise ValueError(f"No mapping found for {code} in {adjustment_period}")
            else:
                # Determine the expected product code length
                product_level = self.get_source_product_level(missing_year)

                missing_code_list = self.expand_partial_code(
                    missing_classification_type,
                    missing_classification,
                    missing_code, 
                    product_level, 
                    adjustment_period, 
                    concordance
                )

                if missing_code_list is None:
                    missing_code_list = [code]

            for missing_code in missing_code_list:
                if missing_classification_type == "source":
                    concorded_products.append({
                        'code.before': missing_code,
                        'code.after': code,
                        'adjustment': adjustment_period
                    })
                else:
                    concorded_products.append({
                        'code.before': code,
                        'code.after': missing_code,
                        'adjustment': adjustment_period
                    })

        if concorded_products:
            new_rows_df = pd.DataFrame(concorded_products)
            new_df =  pd.concat([concordance, new_rows_df], ignore_index=True)
            new_df.drop_duplicates(subset=["code.before", "code.after", "adjustment"], inplace=True)
            return new_df
        return concordance