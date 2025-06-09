import pandas as pd


def clean_groups(groups, source_class, target_class):
    groups = groups.copy()
    # prep groups files
    try:
        groups = groups.drop(columns='Unnamed: 0')
    except:
        pass
    groups['group.id'] = groups['group.id'].astype(int)
    groups['code.source'] = groups['code.source'].astype(str)
    groups['code.target'] = groups['code.target'].astype(str)

    source_detailed_product_level = get_detailed_product_level(source_class)
    target_detailed_product_level = get_detailed_product_level(target_class)
    
    groups.loc[groups['code.source'].str.len() < source_detailed_product_level, 'code.source'] = groups['code.source'].str.zfill(source_detailed_product_level)
    groups.loc[groups['code.target'].str.len() < target_detailed_product_level, 'code.target'] = groups['code.target'].str.zfill(target_detailed_product_level)
    return groups


def get_detailed_product_level(classification):
    if classification.startswith("H"):
        return 6
    elif classification.startswith("S"):
        return 4

