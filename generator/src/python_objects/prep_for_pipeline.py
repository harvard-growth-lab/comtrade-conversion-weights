import glob
import re
import pandas as pd
from src.utils.util import clean_groups


class PipelineWeightPrep:
    def __init__(self, conversion_years):
        self.conversion_years = conversion_years

    def run(self):
        for source_class, start_year, target_class, end_year in self.conversion_years:
            combined_result = pd.DataFrame()
            results = glob.glob(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/conversion_weights/conversion.weights.start.{start_year}.end.{end_year}.group.*.csv")
            for file in results:
                match = re.search(r'group\.(\d+)\.csv$', str(file))
                if not match:
                    continue

                gid = match.group(1)

                # try:
                conversion_group = pd.read_csv(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/matrices/conversion.matrix.start.{start_year}.end.{end_year}.group.{gid}.csv",
                                            dtype={'code.source': str})

                # Load weights and conversion matrix
                weights = pd.read_csv(file, header=None)

                if source_class.startswith("H"):
                    detailed_product_level = 6
                else:
                    detailed_product_level = 4
                # Standardize source product codes
                conversion_group['code.source'] = conversion_group['code.source'].apply(
                    lambda x: x.zfill(detailed_product_level) if len(x) < detailed_product_level and x != "TOTAL" else x
                )

                conversion_group = conversion_group.set_index('code.source')            
                weight_df = pd.DataFrame(weights.values, index=conversion_group.index,columns=conversion_group.columns)

                # Convert to long format
                weight_long = weight_df.reset_index().melt(
                    id_vars='code.source',var_name='code.target',value_name='weight'
                ).astype({'code.source': str, 'code.target':str, 'weight': float})

                weight_long['group_id'] = gid
                combined_result = pd.concat([combined_result, weight_long])
                # except:
                #     print("failed")
                    
            groups = pd.read_csv(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/concordance_groups/from_{source_class}_to_{target_class}.csv", 
                                dtype={source_class: str, target_class: str})
            matched, cleaned_group = clean_groups(groups, source_class, target_class)

            matched = matched[['group.id','code.source','code.target']]
            matched['weight'] = 1
            matched = matched.rename(columns={"group.id":"group_id"})
            combined_result = pd.concat([combined_result, matched])
            
            combined_result[['code.target','code.source']] = combined_result[['code.target','code.source']].astype(str)
            combined_result = combined_result.rename(columns={"code.target":target_class,"code.source":source_class})
                    
            print(f"saving {source_class}:{target_class}.csv")
            combined_result.to_csv(f"/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator/data/output/grouped_weights/grouped_{source_class}:{target_class}.csv", index=False)       
