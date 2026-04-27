"""
Python replacement for src/R_code/create_product_groups.R

Creates product groups: identifies codes linked through classification
correspondence tables and assigns each connected component a group.id.

Consolidated correlation CSVs always use:
  code.after  = NEWER classification code (e.g. NAICS2012)
  code.before = OLDER classification code (e.g. NAICS2007)
  Relationship = "1:1" | "n:1" | "1:n" | "n:n"
                 notation: {#unique code.before} : {#unique code.after} per code
  adjustment  = "<newer_year> to <older_year>"

Direction determines which relationships require group optimization:

  BACKWARD (newer→older, e.g. NAICS2012→NAICS2007):
    • n:1  (n OLDER → 1 NEWER): each older code maps independently to exactly
           one newer code; backward split weights are estimated without joint
           optimization → group.id = NA
    • 1:n  (1 OLDER → n NEWER): backward requires allocating one older code's
           value across multiple newer codes; weights are jointly constrained
           → group.id assigned
    • n:n  (n OLDER → n NEWER): fully interdependent → group.id assigned

  FORWARD (older→newer, e.g. NAICS2007→NAICS2012):
    • 1:n  (1 OLDER → n NEWER): each newer code comes from one older source;
           forward weights estimated independently → group.id = NA
    • n:1  (n OLDER → 1 NEWER): forward requires combining multiple older
           codes into one newer code; weights are jointly constrained
           → group.id assigned
    • n:n  → group.id assigned

  1:1 rows always have group.id = NA.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_COL_AFTER = "code.after"    # newer classification code
_COL_BEFORE = "code.before"  # older classification code
_COL_REL = "Relationship"
_COL_ADJ = "adjustment"
_COL_GROUP = "group.id"

# Relationships requiring joint optimization per conversion direction
_NEEDS_GROUP = {
    "backward": {"1:n", "n:n"},
    "forward":  {"n:1", "n:n"},
}


def create_groups(data: pd.DataFrame) -> pd.DataFrame:
    """
    Find connected components in a code correspondence table.

    Replicates the R create.groups() algorithm: iterates through ungrouped
    rows, expands each seed to its full connected component via transitive
    closure, and assigns sequential integer group.ids.

    Args:
        data: Two-column DataFrame [first_col, second_col] of code pairs
              that need joint optimization.  Column names are preserved.
    Returns:
        DataFrame with the original two columns plus an integer 'group.id';
        every row is assigned (no NA values in the returned frame).
    """
    base = data.drop_duplicates().reset_index(drop=True).copy()
    first_col, second_col = base.columns[0], base.columns[1]
    base[_COL_GROUP] = pd.array([pd.NA] * len(base), dtype=pd.Int64Dtype())

    group_id = 1

    while base[_COL_GROUP].isna().any():
        seed = base.loc[base[_COL_GROUP].isna(), first_col].iloc[0]

        matches_first: set = set(base.loc[base[first_col] == seed, second_col])
        matches_second: set = set(
            base.loc[base[second_col].isin(matches_first), first_col]
        )

        base.loc[base[second_col].isin(matches_first), _COL_GROUP] = group_id
        base.loc[base[first_col].isin(matches_second), _COL_GROUP] = group_id

        prev_first: set = set()
        prev_second: set = set()

        while prev_first != matches_first or prev_second != matches_second:
            prev_first = matches_first.copy()
            prev_second = matches_second.copy()

            matches_first = set(
                base.loc[base[first_col].isin(matches_second), second_col]
            )
            matches_second = set(
                base.loc[base[second_col].isin(matches_first), first_col]
            )

            base.loc[base[second_col].isin(matches_first), _COL_GROUP] = group_id
            base.loc[base[first_col].isin(matches_second), _COL_GROUP] = group_id

        group_id += 1

    return base


def _get_adjustment_string(source_class: str, target_class: str) -> str:
    """
    Derive the adjustment filter from classification names.
    Always '<newer_year> to <older_year>' regardless of conversion direction.
    """
    source_year = int("".join(c for c in source_class if c.isdigit()))
    target_year = int("".join(c for c in target_class if c.isdigit()))
    newer = max(source_year, target_year)
    older = min(source_year, target_year)
    return f"{newer} to {older}"


def _process_pair(
    all_vintages: pd.DataFrame,
    source_class: str,
    target_class: str,
    direction: str,
) -> pd.DataFrame:
    """
    Assign group IDs for one source→target conversion pair.

    Only relationships that require joint optimization receive a group.id
    (see module docstring).  All other rows have group.id = NA.

    Args:
        all_vintages: Full consolidated correlation table.
        source_class:  e.g. "NAICS2012"
        target_class:  e.g. "NAICS2007"
        direction:     "backward" (newer→older) or "forward" (older→newer)

    Returns:
        DataFrame with columns [code.source, code.target, Relationship,
        adjustment, group.id].  Rows that do not require joint optimization
        have group.id = NA.
    """
    if direction not in _NEEDS_GROUP:
        raise ValueError(f"direction must be 'backward' or 'forward', got {direction!r}")

    adjustment = _get_adjustment_string(source_class, target_class)
    conversion = all_vintages[all_vintages[_COL_ADJ] == adjustment].copy()
    conversion[_COL_AFTER] = pd.to_numeric(conversion[_COL_AFTER])
    conversion[_COL_BEFORE] = pd.to_numeric(conversion[_COL_BEFORE])

    # Rows that require joint optimization for this direction
    needs_group_mask = conversion[_COL_REL].isin(_NEEDS_GROUP[direction])

    if needs_group_mask.any():
        groups = create_groups(
            conversion.loc[needs_group_mask, [_COL_AFTER, _COL_BEFORE]]
        )
        conversion = conversion.merge(
            groups, on=[_COL_AFTER, _COL_BEFORE], how="left"
        )
    else:
        conversion[_COL_GROUP] = pd.array(
            [pd.NA] * len(conversion), dtype=pd.Int64Dtype()
        )

    # Rename code.after / code.before to code.source / code.target
    if direction == "backward":
        # newer code (code.after) is the source; older code (code.before) is the target
        conversion = conversion.rename(
            columns={_COL_AFTER: "code.source", _COL_BEFORE: "code.target"}
        )
    else:
        # forward: older code (code.before) is the source; newer (code.after) is the target
        conversion = conversion.rename(
            columns={_COL_BEFORE: "code.source", _COL_AFTER: "code.target"}
        )

    return conversion


class CreateProductGroups:
    """
    Python replacement for src/R_code/create_product_groups.R.

    Reads the consolidated correlation CSV produced by NaicsIngest or
    CombineCorrelationTables and writes one CSV per enabled conversion pair
    into data/correlation_groups/.
    """

    def __init__(
        self,
        conversion_weights_pairs: list,
        data_source: str,
        root_dir: Path,
    ):
        self.conversion_weights_pairs = conversion_weights_pairs
        self.data_source = data_source
        self.root_dir = Path(root_dir)
        self.data_path = self.root_dir / "data"
        self.correlation_groups_path = self.data_path / "correlation_groups"
        self.correlation_groups_path.mkdir(parents=True, exist_ok=True)

        correlation_filename = (
            "consolidated_naics_correlation_tables.csv"
            if data_source == "naics"
            else "consolidated_comtrade_correlation_tables.csv"
        )
        self.correlation_path = (
            self.data_path
            / "output"
            / "consolidated_correlation"
            / correlation_filename
        )

    @staticmethod
    def _get_adjustment_string(source_class: str, target_class: str) -> str:
        return _get_adjustment_string(source_class, target_class)

    def _process_pair(
        self,
        all_vintages: pd.DataFrame,
        source_class: str,
        target_class: str,
        direction: str,
    ) -> pd.DataFrame:
        return _process_pair(all_vintages, source_class, target_class, direction)

    def run(self) -> None:
        all_vintages = pd.read_csv(self.correlation_path)

        for pair in self.conversion_weights_pairs:
            source_class = pair["source_class"]
            target_class = pair["target_class"]
            direction = pair["direction"]

            result = self._process_pair(
                all_vintages, source_class, target_class, direction
            )

            output_path = (
                self.correlation_groups_path
                / f"from_{source_class}_to_{target_class}.csv"
            )
            result.to_csv(output_path, index=False, na_rep="NA")
            logger.info(
                "Generated group assignments: %s → %s  (%d groups, written to %s)",
                source_class,
                target_class,
                result[_COL_GROUP].dropna().nunique(),
                output_path.name,
            )
