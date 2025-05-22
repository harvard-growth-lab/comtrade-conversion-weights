import pandas as pd
import glob


class CombineConcordances():
    def __init__(self, concordance_files):
        self.concordance_files = concordance_files

    def combine_concordances(self):
        pass

    def concatentate_concordance_to_main(self, classification_group="HS"):
        """ 
        RUN TO ADD Comtrade Concordance Tables to A Consolidated/Clean concordance table
        """

        df = pd.DataFrame(columns=['code.after','code.before','Relationship','adjustment'])
        dtype_dict = {'code.after': str, 'code.before': str}
        if classification_group == "SITC":
            # overwrites previous file
            df.to_csv("comtrade_concordance/SITC_consolidated_comtrade_concordances.csv", index=False)
        else:
            df.read_csv(f"comtrade_concordance/{classification_group}_consolidated_comtrade_concordances.csv", index=False)

        concordance_files = glob.glob("comtrade_concordance/*.xls")
        for file in concordance_files:
            source, target = self.extract_classifications(file.split('/')[-1])
            if source.startswith("H") and target.startswith("H"):
                continue
            print(f"concatenating file {source} to {target}")
            # corr = pd.read_excel(file, sheet_name = f"{source}-{target} Correlations", header=0)
            corr = pd.read_excel(file, sheet_name = f"Correlation Tables", header=1, dtype=str)

            df = format_concordance_table(corr)
            if df.empty:
                print(f"already concatenated to file")
                continue

            for col in ['code.after', 'code.before']:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            print(f"added to consolidated concordance table")
            df['Relationship'] = df.Relationship.str.replace(' to ', ':')
            df['code.before'] = df['code.before'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df['code.after'] = df['code.after'].apply(lambda x: x[:-1] if len(x) == 5 else x)
            df = df.drop_duplicates(subset=["code.after", "code.before", "adjustment"])
            if classification_group == "SITC":
                df = self.determine_relationship(df)

            main_df = pd.read_csv(f"comtrade_concordance/{classification_group}_consolidated_comtrade_concordances.csv", dtype={'code.before': str, 'code.after': str})

            if df.adjustment.unique() in main_df.adjustment.unique():
                print(f"already concatenated to file")
                continue

            df = pd.concat([df, main_df])
            df.to_csv(f"comtrade_concordance/{classification_group}_consolidated_comtrade_concordances.csv", index=False)


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
