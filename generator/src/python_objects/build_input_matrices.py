# RUN after Groups from R code has been generated in the data/concordance_groups folder
# PREP CONVERSION, TRADE VALUE TABLES AND WEIGHT TABLES

import pandas as pd
import glob
import os
from datetime import datetime
import re
import comtradeapicall
from pathlib import Path
from src.utils.util import clean_groups, get_detailed_product_level

pd.options.mode.copy_on_write = True
pd.set_option('future.no_silent_downcasting', True)

class MatrixBuilder():
    atlas_classifications = ["H0", "H4", "S1", "S2"]
    RELEASE_YEARS = {"S1" : 1962, "S2": 1976, "S3": 1988, "S4": 2007,
                     "H0": 1995, "H1": 1996, "H2": 2002, "H3": 2007, 
                     "H4": 2012, "H5": 2017, "H6": 2022}


    def __init__(self, weight_tables):
        self.weight_tables = weight_tables
        self.build()
    

    def build(self):
        """
        generates conversion and trade values matrices that is 
        ready for matlab code to generate conversion weights
        """
        raw_parquet_path = "/n/hausmann_lab/lab/atlas/data/as_reported/raw_parquet/"
        aggregated_by_year_path = "/n/hausmann_lab/lab/atlas/data/as_reported/aggregated_by_year/parquet/"
        for direction, source_class, target_class in self.weight_tables:
            print(f"beginning conversion for {source_class} to {target_class}")

            if direction == "backward":
                # H1 => H0, source 1995 & target 1996
                source_year = self.RELEASE_YEARS[source_class]
                target_year = source_year - 1
            else: 
                # H0 => H1, source 1995 & target 1996
                target_year = self.RELEASE_YEARS[target_class]
                source_year = target_year - 1

            avg_range = 3
            if source_class in ["H0", "S3"] and target_class in ["H0", "S3"]:

                if source_class == "H0":
                    source_year = 1992
                    target_year = 1988
                else:
                    source_year = 1988
                    target_year = 1992

                files_target = []
                files_source = []
                for year in range(target_year, target_year + avg_range):
                    files_target += glob.glob(os.path.join(raw_parquet_path, target_class, str(year), "*.parquet"))
                for year in range(source_year - 1, source_year + 2):
                    files_source += glob.glob(os.path.join(raw_parquet_path, source_class, str(year), "*.parquet"))
            
            else:
                files_target = glob.glob(os.path.join(raw_parquet_path, target_class, str(target_year), "*.parquet"))
                files_source = glob.glob(os.path.join(raw_parquet_path, source_class, str(source_year), "*.parquet"))

            print(f"{target_class}: {target_year} to {source_class}: {source_year}")
            files_target = get_files_by_classification_in_year(files_target, target_class)
            files_source = get_files_by_classification_in_year(files_source, source_class)
            comtrade_dict = {target_year: files_target, source_year: files_source}
            reporters = extract_reporters_with_timely_classification_update(comtrade_dict)
            groups = pd.read_csv(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/concordance_groups/from_{source_class}_to_{target_class}.csv")
            if not groups[((groups['code.source'].isna()) | (groups['code.target'].isna()))].empty:
                raise ValueError(f"check the concordance group file for {source_class} to {target_class} \n {groups[((groups['code.source'].isna()) | (groups['code.target'].isna()))]}")
            groups = groups.astype({'code.source':int, 'code.target':int}).astype({'code.source': str, 'code.target': str})
            print(f"There are {len(reporters)} reporters who switched timely from {target_class} to {source_class}")

            # need reporting importer
            print(f"data for {target_class}/{target_class}_{target_year}")
            print(f"data for {source_class}/{source_class}_{source_year}")

            if source_class in ["H0", "S3"] and target_class in ["H0", "S3"]:
                target_df = generate_year_avgs(target_class, target_year, avg_range)
                source_df = generate_year_avgs(source_class, source_year, avg_range)

            else: 
                target_path = os.path.join(aggregated_by_year_path, target_class, f"{target_class}_{target_year}.parquet")
                source_path = os.path.join(aggregated_by_year_path, source_class, f"{source_class}_{source_year}.parquet")
                # CPY for reporting imports
                target_df = pd.read_parquet(target_path)
                source_df = pd.read_parquet(source_path)

            # only want timely reporters switch to new classification upon release
            target_df = filter_df_for_reporters(target_class, target_df, reporters)
            source_df = filter_df_for_reporters(source_class, source_df, reporters)

            # extract 1:N, N:1 , N:N relationships 
            _, groups = clean_groups(groups, source_class, target_class)
            if 1 in groups.groupby("group.id").agg({"group.id":"count"})['group.id'].unique():
                raise ValueError(f"grouping of one product is invalid.")
            if "1:1" in groups.Relationship.unique():
                raise ValueError(f"grouping of one to one relationship, is invalid.")

            target_dfs = country_by_prod_trade(target_df, groups, "target", target_class)
            source_dfs = country_by_prod_trade(source_df, groups, "source", source_class)

            target_dfs, source_dfs = align_reporter_indices(groups, target_dfs, source_dfs)

            group_dfs = conversion_matrix(groups)

            generate_dataframes(target_dfs, "target.trade", source_year, target_year)
            generate_dataframes(source_dfs, "source.trade", source_year, target_year)
            generate_dataframes(group_dfs, "conversion", source_year, target_year)


def generate_dataframes(dfs, table, source_year, target_year):
        # generate data frames 

    files = glob.glob(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/matrices/{table}.matrix.start.{source_year}.end.{target_year}.group.*.csv")

    # clean out previously generated files
    for file_path in files:
        try:
            # Check if file exists before attempting to delete
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

    for group_id, df in dfs.items():
        if df.empty:
            # drop nans
            print(f"df is empty for {table} group: {group_id}")
            continue
        
        df = df.fillna(0)
        os.makedirs(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/matrices", exist_ok=True)
        df.to_csv(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/matrices/{table}.matrix.start.{source_year}.end.{target_year}.group.{group_id}.csv")

        
def align_reporter_indices(groups, target_dfs, source_dfs):
    # enforces shared reporter indices 
    for group_id in groups['group.id'].unique().tolist():
        tdf = target_dfs[group_id].reset_index()
        sdf = source_dfs[group_id].reset_index()
        target_dfs[group_id] = tdf.merge(sdf[['reporterISO3']], on='reporterISO3', how='outer').set_index('reporterISO3').sort_index()
        source_dfs[group_id] = sdf.merge(tdf[['reporterISO3']], on='reporterISO3', how='outer').set_index('reporterISO3').sort_index()
        
    return target_dfs, source_dfs

def get_files_by_classification_in_year(files, classification_source):
    files_classification_year = []
    files = [file.split('/')[-1] for file in files]
    for file in files:
        f = ComtradeFile(file)
        if f.classification == classification_source:
            files_classification_year.append(file)
    # print(files_classification_year)
    return files_classification_year

def extract_reporters_with_timely_classification_update(comtrade_dict):
    reporters_dict = {}
    for year, files in comtrade_dict.items():
        for file in files:
            f = ComtradeFile(file)
            reporters_dict[str(f.reporter_code)] = reporters_dict.get(str(f.reporter_code), []) + [file]
    return [key for key, value in reporters_dict.items() if isinstance(value, list) and len(value) >= 2]


def generate_year_avgs(classification, start_year, avg_range):
    df = pd.DataFrame(columns=['reporterISO3', 'partnerISO3',  'cmdCode', "qty", "CIFValue", "FOBValue", "primaryValue"])
    
    detailed_product_level = get_detailed_product_level(classification)

    print(f"adding starting year {start_year} into the avg")    
    df_path = f"/n/hausmann_lab/lab/atlas/data/as_reported/aggregated_by_year/parquet/{classification}/{classification}_{start_year}.parquet"
    df = pd.read_parquet(df_path)

    df = df[(df.flowCode=="M")&(df.digitLevel==detailed_product_level)]
    df = df.groupby(['reporterISO3', 'partnerISO3',  'cmdCode']).agg({"qty":"sum","CIFValue":"sum", "FOBValue": "sum", "primaryValue": "sum"}).reset_index()
    df = df.rename(columns={"qty":f"qty_{start_year}","CIFValue":f"CIFValue_{start_year}", "FOBValue": f"FOBValue_{start_year}", "primaryValue": f"primaryValue_{start_year}"})
    
    for year in range(start_year + 1, start_year + avg_range):
        print(f"adding {year} into the avg")
        df_path = f"/n/hausmann_lab/lab/atlas/data/as_reported/aggregated_by_year/parquet/{classification}/{classification}_{year}.parquet"
        single_year = pd.read_parquet(df_path)
        single_year = single_year[(single_year.flowCode=="M")&(single_year.digitLevel==detailed_product_level)]
        single_year = single_year.groupby(['reporterISO3', 'partnerISO3',  'cmdCode']).agg({"qty":"sum","CIFValue":"sum", "FOBValue": "sum", "primaryValue": "sum"}).reset_index()

        df = single_year.merge(df, on=['reporterISO3', 'partnerISO3',  'cmdCode'],
                               how='outer',
                               suffixes=(f"_{year}", ""))
    
    for trade_value in ["FOBValue", "primaryValue", "CIFValue", "qty"]:
        matching_columns = [col for col in df.columns if trade_value in col]
        df[f"{trade_value}_avg"] = df[matching_columns].mean(axis=1)
        df = df.drop(columns=matching_columns)
        df = df.rename(columns={f"{trade_value}_avg":trade_value})

    df['flowCode'] = "M"
    df['digitLevel'] = detailed_product_level
    return df
        

def filter_df_for_reporters(classification, df, reporters):
    detailed_product_level = get_detailed_product_level(classification)

    df = df[(df.flowCode=="M")&(df.digitLevel==detailed_product_level)]
    reporter = comtradeapicall.getReference("reporter")
    partner = comtradeapicall.getReference("partner")
    reporter = reporter.astype({'reporterCode': str})
    partner = partner.astype({'PartnerCode': str})
    partner.loc[partner['PartnerCodeIsoAlpha3']=="W00", 'PartnerCodeIsoAlpha3'] = "WLD"
    
    df = df.merge(reporter[['reporterCode','reporterCodeIsoAlpha3']], left_on=['reporterISO3'], right_on="reporterCodeIsoAlpha3", how="left")
    df = df.merge(partner[['PartnerCode','PartnerCodeIsoAlpha3']], left_on=['partnerISO3'], right_on="PartnerCodeIsoAlpha3", how="left")
    
    return df[df.reporterCode.isin(reporters)]


def country_by_prod_trade(df, groups, classification_type, prod_class):
    dfs = {}
    for group_id in groups['group.id'].unique():
        group = groups[groups['group.id']==group_id].copy()
        detailed_product_level = get_detailed_product_level(prod_class)
        if classification_type == "target":
            group.loc[:, 'code.target'] = group['code.target'].astype(str)
            
            group.loc[group['code.target'].str.len() < detailed_product_level, 'code.target'] = group['code.target'].str.zfill(detailed_product_level)
            
            product_codes = group['code.target'].unique().tolist()
            filtered_df = df[df.cmdCode.isin(product_codes)]
            if filtered_df.empty:
                import pdb; pdb.set_trace()
        elif classification_type == "source":
            group.loc[:, 'code.source'] = group['code.source'].astype(str)
            group.loc[group['code.source'].str.len() < detailed_product_level, 'code.source'] = group['code.source'].str.zfill(detailed_product_level)
            
            product_codes = group['code.source'].unique().tolist()
            filtered_df = df[df.cmdCode.isin(product_codes)]
            if filtered_df.empty:
                import pdb; pdb.set_trace()

        pivot_df = filtered_df.pivot_table(values='primaryValue', index='reporterISO3',columns='cmdCode')
        # pivot_df.fillna(0)
        dfs[group_id] = pivot_df
    return dfs


def conversion_matrix(groups):
    # by group
    # rows are the source
    # cols are the target
    dfs = {}
    for group_id in groups['group.id'].unique():
        group = groups[groups['group.id']==group_id]
        df = group.pivot_table(values='group.id', index='code.source',columns='code.target')
        df = df.replace(group_id, True).infer_objects(copy=False)
        df = df.fillna(False)
        dfs[group_id] = df
    return dfs


def extract_classifications(filename):
    # Pattern to match classifications: letters followed by optional digits
    pattern = r'([A-Za-z]+\d*)'
    
    # Find all matches
    matches = re.findall(pattern, filename)
    
    # Filter out non-classification words like "to", "Conversion", etc.
    classifications = [m for m in matches if m not in ["to", "Conversion", "and", "Correlation", "Tables", "xls"]]
    source, target = classifications[0], classifications[1]
    return source, target


def determine_relationship(concordance_df):
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


class ComtradeFile:
    """Parses and stores Comtrade file metadata."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.name = self.file_path.name
        self._parse_filename()

    def _parse_filename(self) -> None:
        patterns = [
            r"COMTRADE-FINALCLASSIC-CA(?P<reporter>\d{3})(?P<year>\d{4})(?P<classification>\w+)\[(?P<date>[\d-]+)\]",
            r"COMTRADE-FINAL-CA(?P<reporter>\d{3})(?P<year>\d{4})(?P<classification>\w+)\[(?P<date>[\d-]+)\]",
        ]

        for pattern in patterns:
            match = re.match(pattern, self.name)
            if match:
                self.match = match
                self.reporter_code = match.group("reporter")
                self.year = int(match.group("year"))
                self.classification = match.group("classification")
                self.published_date = datetime.strptime(match.group("date"), "%Y-%m-%d")
                return
        raise ValueError(f"File format has not been handled: {self.name}")
        
        
    def swap_classification(self, new_classification):
        """
        Swap the classification in the filename and update the object's properties.

        Parameters:
        -----------
        new_classification : str
            The new classification code to use (e.g., 'H0', 'H1', 'S3', etc.)

        Returns:
        --------
        ComtradeFile
            Returns self for method chaining
        """
        old_filename = self.name

        self.classification = new_classification

        if "FINALCLASSIC" in old_filename:
            pattern = r"(COMTRADE-FINALCLASSIC-CA\d{3}\d{4})(\w+)(\[[\d-]+\]\.parquet)"
        else:
            pattern = r"(COMTRADE-FINAL-CA\d{3}\d{4})(\w+)(\[[\d-]+\]\.parquet)"

        # Replace the classification part with the new classification
        self.name = re.sub(pattern, fr"\1{new_classification}\3", old_filename)

        # Update the file_path to match the new name
        self.file_path = self.file_path.parent / self.name

        return self


class ComtradeFiles:
    """Extract file(s) paths based on provided metadata"""

    def __init__(self, files):
        self.files = files

    def get_file_names(self, reporter_code, dates) -> list:
        files = set()
        for f in self.files:
            for date in dates:
                date_str = date.strftime("%Y-%m-%d")
                file = re.search(
                    f".*COMTRADE-FINAL-CA{reporter_code}\\d{{4}}\\w+\\[{date_str}]", f
                )
                try:
                    files.add(file.string)
                except AttributeError as e:
                    pass
        return files

        

        
                 

