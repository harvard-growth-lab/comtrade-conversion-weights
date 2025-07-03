import pandas as pd
import glob
import itertools
import re
from typing import Dict, List, Tuple, Set
import numpy as np
from scipy.sparse import csr_matrix
from src.utils.util import get_detailed_product_level



class CombineConcordances():

    SITC_DETAIL_PRODUCT_CODE_LENGTH = 4
    HS_DETAIL_PRODUCT_CODE_LENGTH = 6
    SITC_YEAR_CUTOFF = 1988
    MAX_TRUNCATION_ATTEMPTS = 4
    # non_concorded_product_file = "data/static/product_missing_concordance.csv"


    RELEASE_YEARS = {
        "SITC1" : 1962, "SITC2": 1976, "SITC3": 1988, 
                     "H0": 1992, "H1": 1996, "H2": 2002, 
                     "H3": 2007, "H4": 2012, "H5": 2017, "H6": 2022
                     }


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
        for file in concordance_files:
            source, target = self.extract_classifications(file.split('/')[-1])
            df = pd.read_excel(file, sheet_name = f"Correlation Tables", header=1, dtype=str)
            df = self.format_concordance_table(df)

            for col in ['code.after', 'code.before']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            print(f"added {file.split('/')[-1]} to consolidated concordance table")
            non_concorded_products = []
            non_concorded_df = pd.DataFrame()
            non_concorded_df.to_csv(f"data/output/non_concorded_products.csv", index=False, mode='a')
            for classification_type, classification in [("source", source), ("target", target)]:
                print(f"loading in {classification} products")
                products = pd.read_excel(f"data/static/all_products_by_classification/{classification}.xlsx", sheet_name="Sheet1", dtype=str)
                non_concorded_df = self.find_non_concorded_products(classification_type, classification, df, products)
                non_concorded_products.append(non_concorded_df)
            non_concorded_df = pd.concat(non_concorded_products)
            non_concorded_df.to_csv(f"data/output/non_concorded_products.csv", index=False, mode='a')
            df['code.before'] = df['code.before'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df['code.after'] = df['code.after'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df = self.handle_no_concordances(df, non_concorded_df)
            
            df = df.drop_duplicates(subset=["code.after", "code.before", "adjustment"])
            df = self.handle_special_cases(df)
            df = self.recalculate_relationship_column(df)
            dfs.append(df)

        consolidated_df = pd.concat(dfs)
        consolidated_df = consolidated_df[~consolidated_df['code.before'].isin(["I", "II"])]
        consolidated_df = consolidated_df[~((consolidated_df['code.before']=='nan') & (consolidated_df['code.after']=='nan'))]
        consolidated_df = consolidated_df[~((consolidated_df['code.before'].isna()) | (consolidated_df['code.after'].isna()))]
        consolidated_df.to_csv(f"data/output/consolidated_concordance/consolidated_comtrade_concordances.csv", index=False)


    def extract_classifications(self,filename):
        # Pattern to match classifications: letters followed by optional digits
        years = re.findall(r'\b\d{4}\b', filename)
        if years:
            source_classification = [k for k, v in self.RELEASE_YEARS.items() if v == int(years[0])][0]
            target_classification = [k for k, v in self.RELEASE_YEARS.items() if v == int(years[1])][0]
            return source_classification, target_classification
        else:
            pattern = r'([A-Za-z]+\d*)'
            matches = re.findall(pattern, filename)
            classifications = [m for m in matches if m not in ["to", "Conversion", "and", "Correlation", "Tables", "xls"]]
            return classifications[0], classifications[1]
    
    def recalculate_relationship_column(self, df):
        """
        Analyze and classify relationships between product codes using Conditional Independence Filter (CIF) analysis.
        
        This method implements the first step of CIF calculations by analyzing the relationships between
        product codes in different classification systems. It creates a sparse matrix representation of
        product code mappings and classifies relationships into four categories:
        
        - '1:1': One-to-one mapping between source and target codes
        - 'n:1': Many-to-one mapping (multiple source codes map to one target code)
        - '1:n': One-to-many mapping (one source code maps to multiple target codes)
        - 'n:n': Many-to-many mapping (multiple source codes map to multiple target codes)
        
        The analysis is performed separately for each adjustment period to ensure temporal consistency.
        
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
            
        Notes
        -----
        This is a key component of the CIF methodology that helps identify which product
        code pairs require optimization of conversion weights. Only non-1:1 relationships
        are considered for subsequent optimization steps.
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


    def expand_partial_code(self, missing_classification_type, missing_classification, matched_code, product_level, adjustment_period, concordance):
        """
        Expand a partial source code to all matching detailed codes.
        """
        col = "code.after" if missing_classification_type == "source" else "code.before"
        if len(matched_code) < product_level:
            all_products = pd.read_excel(f"data/static/all_products_by_classification/{missing_classification}.xlsx", sheet_name="Sheet1", dtype=str)
            matching_codes = all_products[(all_products.id.str.startswith(matched_code))&(all_products.id.str.len()==product_level)]['id'].unique().tolist()
            matching_codes = [code for code in matching_codes if code.isdigit()]
            return matching_codes if matching_codes else None
        return [matched_code]
    
    def find_non_concorded_products(self, classification_type, classification, concordance, products):
        """
        Find products that are not concorded in the concordance table
        """
        detailed_product_level = get_detailed_product_level(classification)
        missing_products = []
        adjustment_period = concordance.adjustment.unique().tolist()[0]
        if classification_type == "source":
            col = "code.after"
            release_year = adjustment_period.split(" to ")[0]
        elif classification_type == "target":
            col = "code.before"
            release_year = adjustment_period.split(" to ")[1]
        products = products[products.aggrlevel == f"{detailed_product_level}"]
        # drop products with non digits characters
        products = products[products['id'].str.isdigit()]
        missing_products_df = products[~products['id'].str[:detailed_product_level].isin(concordance[col].str[:detailed_product_level].unique().tolist())]

        # handle special cases
        # missing_products_df = self.handle_special_cases(missing_products_df, adjustment_period)

        new_columns = {
            'adjustment': adjustment_period,
            'missing_classification_type': classification_type,
            'missing_year': release_year,
            'missing_classification': classification
        }
        missing_products_df = missing_products_df.assign(**new_columns)
        return missing_products_df[['adjustment','id','text','missing_classification_type', 'missing_year', 'missing_classification']]

    # def handle_special_cases(self, missing_products_df, adjustment_period):
    #     """
    #     Handle special cases where the product code is not in the concordance table
    #     """
    #     # trade not reported
    #     skip_products = []
    #     if adjustment_period == "1992 to 1988":
    #         skip_products = ["999999", "9999AA"]
    #     elif adjustment_period == "2002 to 1996":
    #         skip_products = ["2710"]
    #     return missing_products_df[~missing_products_df['id'].isin(skip_products)]

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
            # if row.adjustment not in non_corded_product_matches:
            #     continue
            missing_classification_type = row.missing_classification_type
            missing_classification = row.missing_classification
            adjustment_period = row.adjustment
            code = str(row.id)
            if missing_classification_type == "source":
                missing_year, matching_year = row.adjustment.split(" to ")[0], row.adjustment.split(" to ")[1]
                matching_classification_type = "target"
                matching_classification = [k for k, v in self.RELEASE_YEARS.items() if v == int(row.adjustment.split(" to ")[1])][0]
            else:
                missing_year, matching_year = row.adjustment.split(" to ")[1], row.adjustment.split(" to ")[0]
                matching_classification_type = "source"
                matching_classification = [k for k, v in self.RELEASE_YEARS.items() if v == int(row.adjustment.split(" to ")[0])][0]
            # Try to find a manual mapping
            matching_code, found = self.find_mapped_source_code(
                code, 
                adjustment_period, 
                non_corded_product_matches
            )

            if not found:
                raise ValueError(f"No mapping found for {code} in {adjustment_period}")
            else:
                # Determine the expected product code length
                product_level = self.get_source_product_level(matching_year)

                matching_code_list = self.expand_partial_code(
                    matching_classification_type,
                    matching_classification,
                    matching_code, 
                    product_level, 
                    adjustment_period, 
                    concordance
                )

                if matching_code_list is None:
                    matching_code_list = [code]

            for matching_code in matching_code_list:
                if missing_classification_type == "source":
                    concorded_products.append({
                        'code.after': code,
                        'code.before': matching_code,
                        'adjustment': adjustment_period
                    })
                else:
                    concorded_products.append({
                        'code.after': matching_code,
                        'code.before': code,
                        'adjustment': adjustment_period
                    })
        if concorded_products:
            newly_concorded_products = pd.DataFrame(concorded_products)
            newly_concorded_products.to_csv(f"data/output/consolidated_concordance/non_concorded_products_matched.csv", index=False, mode='a')
            new_df =  pd.concat([concordance, newly_concorded_products], ignore_index=True)
            new_df.drop_duplicates(subset=["code.before", "code.after", "adjustment"], inplace=True)
            return new_df
        return concordance
    

    def handle_special_cases(self, df):
        """
        Handle special cases where the product code is not in the concordance table
        """
        special_cases = {
            "1992 to 1988":('code.before', ["9999"]),
            "1996 to 1992": ('code.before', ["380993"]),
            "2002 to 1996": ('code.before', ["271091", "271099"]),
            "2002 to 1996": ('code.after', ['271091', '271099']),
            "2002 to 1996": ('code.before', ['271096','271029','271091',
                                            '271022','271094','271095',
                                            '271012','271015','271025',
                                            '271093','271099','271027',
                                            '271016','271026','271021',
                                            '271014','271013']),
        }

        for adjustment_period, (code_type, codes) in special_cases.items():
            if adjustment_period in df.adjustment.unique():
                try:
                    df = df[~df[code_type].isin(codes)]
                except:
                    print(f"Error dropping {codes} for {adjustment_period}")
        return df
