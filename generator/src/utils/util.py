import pandas as pd
from src.python_objects.base import Base

def clean_groups(groups, source_class, target_class):
    groups = groups.copy()
    # prep groups files
    try:
        groups = groups.drop(columns="Unnamed: 0")
    except:
        pass
    groups["code.source"] = groups["code.source"].astype(str)
    groups["code.target"] = groups["code.target"].astype(str)

    source_detailed_product_level = get_detailed_product_level(source_class)
    target_detailed_product_level = get_detailed_product_level(target_class)

    groups.loc[
        groups["code.source"].str.len() < source_detailed_product_level, "code.source"
    ] = groups["code.source"].str.zfill(source_detailed_product_level)
    groups.loc[
        groups["code.target"].str.len() < target_detailed_product_level, "code.target"
    ] = groups["code.target"].str.zfill(target_detailed_product_level)
    return groups


def get_detailed_product_level(classification):
    # NAICS or HS Classification at 6 digit
    if classification.startswith("H") or classification.startswith("N"):
        return 6
    # SITC at 4 digit
    elif classification.startswith("S"):
        return 4
    else:
        raise ValueError(f"Unknown classification: {classification}")


def format_product_code(df: pd.DataFrame, classification: str) -> str:
    """
    Format a product code appropriately for its classification.
    Pads with zeros for Comtrade, no padding for NAICS.
    """
    if classification in Base.HAS_LEADING_ZEROS:
        if Base.HAS_LEADING_ZEROS[classification]:
            detail_level = Base.DETAIL_PRODUCT_CODE_LENGTH[classification]
            return code.zfill(detail_level)
        else:
            return code  # NAICS - no padding
    
    # Fallback for short codes
    if classification.startswith("N"):
        return code
    else:
        detail_level = get_detailed_product_level(classification)
        return code.zfill(detail_level)

def get_classification_family(classification: str) -> str:
    """Returns 'comtrade' or 'naics'"""
    if classification in Base.CLASSIFICATION_FAMILIES:
        return Base.CLASSIFICATION_FAMILIES[classification]
    
    # Fallback
    if classification.startswith("N"):
        return "naics"
    else:
        return "comtrade"

def cleanup_files_from_dir(files):
    for file in files:
        if file.is_file():
            file.unlink()


def cleanup_input_matrices(obj):
    for conversion_weight_pair in obj.conversion_weights_pairs:
        matrices_dir = obj.data_path / "matrices"

        # clear matrices from previous runs if they exist for this conversion
        if matrices_dir.exists():
            matrices_clean_up_files = list(
                matrices_dir.glob(
                    f"conversion.matrix.start.{conversion_weight_pair['source_year']}.end.{conversion_weight_pair['target_year']}.group.*.csv"
                )
            )
            cleanup_files_from_dir(matrices_clean_up_files)


def cleanup_weight_files(obj):
    """
    only deletes conversion weight files for enabled pairs 
    """
    for conversion_weight_pair in obj.conversion_weights_pairs:
        weights_dir = obj.data_path / "conversion_weights"
        if weights_dir.exists():
            conversion_clean_up_files = list(
                weights_dir.glob(
                    f"conversion.weights.start.{conversion_weight_pair['source_year']}.end.{conversion_weight_pair['target_year']}.group.*.csv"
                )
            )
            cleanup_files_from_dir(conversion_clean_up_files)
