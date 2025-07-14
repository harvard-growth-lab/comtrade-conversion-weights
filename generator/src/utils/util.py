import pandas as pd


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
    if classification.startswith("H"):
        return 6
    elif classification.startswith("S"):
        return 4


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
    for conversion_weight_pair in obj.conversion_weights_pairs:
        weights_dir = obj.data_path / "conversion_weights"
        if weights_dir.exists():
            conversion_clean_up_files = list(
                weights_dir.glob(
                    f"conversion.weights.start.{conversion_weight_pair['source_year']}.end.{conversion_weight_pair['target_year']}.group.*.csv"
                )
            )
            cleanup_files_from_dir(conversion_clean_up_files)
