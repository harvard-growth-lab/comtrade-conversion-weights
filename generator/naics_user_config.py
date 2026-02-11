"""
Weight Generator Configuration

This configuration file controls how the weight generator runs.
Edit the settings below to match your environment and requirements.

https://www.census.gov/naics/
"""

from pathlib import Path
import sys
from datetime import date, timedelta
from src.config.source_target_pairs import SOURCE_TARGET_ENABLED_PAIRS


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# top file directory path data downloaded from naics
RAW_DOWNLOADED_COMTRADE_DATA_PATH = (
    "/n/hausmann_lab/lab/atlas/data/"
)

# =============================================================================
# BULK ENABLE/DISABLE SETTINGS
# =============================================================================
# Quick toggles for common scenarios
ENABLE_ALL_FORWARD = False
ENABLE_ALL_BACKWARD = False
ENABLE_ALL_CONVERSIONS = False

# =============================================================================
# CONVERSION PAIR SETTINGS
# =============================================================================
# Enable/disable specific source-target class pairs for conversion
# Set to True for the conversions you want to enable

# optimized weights are provided in the comtrade-downloader as a static data input

# BACKWARD HS CONVERSIONS (newer to older)
CONVERT_NAICS2022_TO_NAICS2017 = True
CONVERT_NAICS2017_TO_NAICS2012 = False
CONVERT_NAICS2012_TO_NAICS2007 = False
CONVERT_NAICS2007_TO_NAICS2002 = False
CONVERT_NAICS2002_TO_NAICS1997 = False
CONVERT_NAICS1997_TO_SIC1987 = False


# FORWARD HS CONVERSIONS (older to newer)
CONVERT_NAICS1997_TO_NAICS2002 = False
CONVERT_NAICS2002_TO_NAICS2007 = False
CONVERT_NAICS2007_TO_NAICS2012 = False
CONVERT_NAICS2012_TO_NAICS2017 = False
CONVERT_NAICS2017_TO_NAICS2022 = True

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "INFO"

# =============================================================================
# PROCESSING STEPS TO RUN (ADVANCED)
# =============================================================================

COMBINE_CONCORDANCES = True
CREATE_PRODUCT_GROUPS = True
BUILD_INPUT_MATRICES = True
GENERATE_WEIGHTS = True
GROUP_WEIGHTS = True

# =============================================================================
# SETUP
# =============================================================================

def get_enabled_conversions():
    """
    Returns a list of enabled conversion pairs based on the settings above.
    """
    enabled_pairs = []

    conversions = [
        (CONVERT_NAICS2022_TO_NAICS2017, "NAICS2022", "NAICS2017", "backward", "2022", "2021"),
        (CONVERT_NAICS2017_TO_NAICS2012, "NAICS2017", "NAICS2012", "backward", "2017", "2016"),
        (CONVERT_NAICS2012_TO_NAICS2007, "NAICS2012", "NAICS2007", "backward", "2012", "2011"),
        (CONVERT_NAICS2007_TO_NAICS2002, "NAICS2007", "NAICS2002", "backward", "2007", "2006"),
        (CONVERT_NAICS2002_TO_NAICS1997, "NAICS2002", "NAICS1997", "backward", "2002", "2001"),
        (CONVERT_NAICS1997_TO_SIC1987, "NAICS1997", "SIC1987", "backward", "1997", "1996"),
        
        (CONVERT_NAICS2017_TO_NAICS2022, "NAICS2017", "NAICS2022", "forward", "2017", "2022"),
        (CONVERT_NAICS2012_TO_NAICS2017, "NAICS2012", "NAICS2017", "forward", "2012", "2017"),
        (CONVERT_NAICS2007_TO_NAICS2012, "NAICS2007", "NAICS2012", "forward", "2007", "2012"),
        (CONVERT_NAICS2002_TO_NAICS2007, "NAICS2002", "NAICS2007", "forward", "2002", "2007"),
        (CONVERT_NAICS1997_TO_NAICS2002, "NAICS1997", "NAICS2002", "forward", "1997", "2002"),
        (CONVERT_SIC1987_TO_NAICS1997, "SIC1987", "NAICS1997", "forward", "1987", "1997"),
    ]

    if ENABLE_ALL_CONVERSIONS:
        for enabled, source, target, direction, source_year, target_year in conversions:
            enabled_pairs.append(
                {
                    "direction": direction,
                    "source_class": source,
                    "target_class": target,
                    "source_year": source_year,
                    "target_year": target_year,
                    "enabled": True,
                }
            )
    else:
        # individually set
        for enabled, source, target, direction, source_year, target_year in conversions:
            # enable bulk settings
            if ENABLE_ALL_FORWARD and direction == "forward":
                enabled = True
            elif ENABLE_ALL_BACKWARD and direction == "backward":
                enabled = True

            if enabled:
                enabled_pairs.append(
                    {
                        "direction": direction,
                        "source_class": source,
                        "target_class": target,
                        "source_year": source_year,
                        "target_year": target_year,
                        "enabled": True,
                    }
                )

    return enabled_pairs
