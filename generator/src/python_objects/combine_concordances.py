import pandas as pd
import glob
import itertools
import re


class CombineConcordances():

    SITC_DETAIL_PRODUCT_CODE_LENGTH = 4
    HS_DETAIL_PRODUCT_CODE_LENGTH = 6
    SITC_YEAR_CUTOFF = 1988
    MAX_TRUNCATION_ATTEMPTS = 4
    non_concorded_product_file = "data/static/product_missing_concordance.csv"

    def __init__(self, classification_group):
        self.classification_group = classification_group
        self.concatentate_concordance_to_main()
        # self.concordance_files = concordance_files


    def concatentate_concordance_to_main(self):
        """ 
        RUN TO ADD Comtrade Concordance Tables to A Consolidated/Clean concordance table
        """

        df = pd.DataFrame(columns=['code.after','code.before','Relationship','adjustment'])
        dtype_dict = {'code.after': str, 'code.before': str}
        if self.classification_group == "SITC":
            # overwrites previous file
            df.to_csv("data/comtrade_concordance/SITC_consolidated_comtrade_concordances.csv", index=False)
        else:
            df = pd.read_csv(f"data/comtrade_concordance/{self.classification_group}_consolidated_comtrade_concordances.csv", index_col=None, dtype={'code.before': str, 'code.after': str})

        concordance_files = glob.glob("data/comtrade_concordance/*.xls")
        for file in concordance_files:
            source, target = self.extract_classifications(file.split('/')[-1])
            if source.startswith("H") and target.startswith("H"):
                continue
            print(f"concatenating file {source} to {target}")
            # corr = pd.read_excel(file, sheet_name = f"{source}-{target} Correlations", header=0)
            corr = pd.read_excel(file, sheet_name = f"Correlation Tables", header=1, dtype=str)

            df = self.format_concordance_table(corr)
            if df.empty:
                print(f"already concatenated to file")
                continue

            for col in ['code.after', 'code.before']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            print(f"added to consolidated concordance table")
            non_concorded_df = pd.read_csv(self.non_concorded_product_file, dtype={"id": str})
            df = self.handle_no_concordances(df, non_concorded_df)
            df['Relationship'] = df.Relationship.str.replace(' to ', ':')
            df['code.before'] = df['code.before'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df['code.after'] = df['code.after'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df = df.drop_duplicates(subset=["code.after", "code.before", "adjustment"])
            # if self.classification_group == "SITC":
            df = self.determine_relationship(df)

            main_df = pd.read_csv(f"data/comtrade_concordance/{self.classification_group}_consolidated_comtrade_concordances.csv", dtype={'code.before': str, 'code.after': str})

            if df.adjustment.unique() in main_df.adjustment.unique():
                print(f"already concatenated to file")
                continue

            df = pd.concat([df, main_df])
            df.to_csv(f"data/comtrade_concordance/{self.classification_group}_consolidated_comtrade_concordances.csv", index=False)


    def extract_classifications(self,filename):
        # Pattern to match classifications: letters followed by optional digits
        pattern = r'([A-Za-z]+\d*)'
        
        # Find all matches
        matches = re.findall(pattern, filename)
        
        # Filter out non-classification words like "to", "Conversion", etc.
        classifications = [m for m in matches if m not in ["to", "Conversion", "and", "Correlation", "Tables", "xls"]]
        source, target = classifications[0], classifications[1]
        return source, target


    def determine_relationship(self,concordance_df):
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
        
        # Create full mappings between codes
        before_to_after_dict = {}
        after_to_before_dict = {}
        
        # Build the mapping dictionaries
        for _, row in df.iterrows():
            before_code = row['code.before']
            after_code = row['code.after']
            
            if before_code not in before_to_after_dict:
                before_to_after_dict[before_code] = set()
            before_to_after_dict[before_code].add(after_code)
            
            if after_code not in after_to_before_dict:
                after_to_before_dict[after_code] = set()
            after_to_before_dict[after_code].add(before_code)
            
        # Determine relationships for each pair
        for i, row in df.iterrows():
            before_code = row['code.before']
            after_code = row['code.after']
            
            before_maps_to_multiple = len(before_to_after_dict[before_code]) > 1
            multiple_map_to_after = len(after_to_before_dict[after_code]) > 1
            
            if not before_maps_to_multiple and not multiple_map_to_after:
                df.at[i, 'Relationship'] = "1:1"
            elif before_maps_to_multiple and not multiple_map_to_after:
                df.at[i, 'Relationship'] = "n:1"
            elif not before_maps_to_multiple and multiple_map_to_after:
                df.at[i, 'Relationship'] = "1:n"
            else:
                df.at[i, 'Relationship'] = "n:n"   
        return df
    

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


    def expand_partial_code(self, missing_classification, code, product_level, adjustment_period, concordance):
        """
        Expand a partial source code to all matching detailed codes.
        """
        col = "code.after" if missing_classification == "source" else "code.before"
        if len(code) < product_level:
            matching_codes = concordance[
                (concordance.adjustment == adjustment_period) & 
                (concordance[col].str.startswith(code))
            ][col].unique().tolist()
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
            missing_classification = row.missing_classification
            adjustment_period = row.adjustment
            code = str(row.id)  # Convert to string for consistency
            if missing_classification == "source":
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
                import pdb; pdb.set_trace()
                raise ValueError(f"No mapping found for {code} in {adjustment_period}")
            else:
                # Determine the expected product code length
                product_level = self.get_source_product_level(missing_year)

                missing_code_list = self.expand_partial_code(
                    missing_classification,
                    missing_code, 
                    product_level, 
                    adjustment_period, 
                    concordance
                )

                if missing_code_list is None:
                    missing_code_list = [code]

            for missing_code in missing_code_list:
                if missing_classification == "source":
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