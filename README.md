# Comtrade Conversion Weights

 Generates conversion weights to harmonize the Standard International Trade Classification (SITC) and Harmonized System (HS) across adjacent vintages 
 
## What This Does

Generates conversion weights between adjacent classification vintages using the Growth Lab Trade Data Methodology by combining UN correspondence tables with observed trade patterns and constrained optimization.

EX: (HS1992 → HS1996, SITC3 → HS1992, etc.) 

![A digram of the processing for the comtrade-conversion-weights. We begin with data as reported by Country Reporters & Correlation Tables from Comtrade. Each adjacent classification pair (adjustment period) in a Correlation Table is disaggrated into groups - which can comprehensively correlates the source classification vintage product codes into the target classification vintage product codes. Next, the trade data from Comtrade is used to identify the set of reporters that switched to reporting in the next classification vintage the year it was released. Matrices grouped by their Correlation grouping of products and the associated trade value in the source and target classification vintage for the timely set of reporters serves as the inputs to the optimization algorithm which outputs the conversion weights. ](generator/images/comtrade_weight_conversion_diagram.png)

The diagram above illustrates the processing steps used to generate conversion weights from Comtrade's correlation tables published by the World Customs Organization (WCO). This methodology underpins the bilateral trade data published in the [Atlas of Economic Complexity](https://atlas.hks.harvard.edu/).


### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/) for managing dependencies
- [R](https://cran.rstudio.com/) for product grouping
- MATLAB R2021a for optimization code
- [Premium UN Comtrade API key](https://comtradeplus.un.org/)
- Raw Comtrade data files (([download from comtrade-downloader](https://github.com/harvard-growth-lab/comtrade-downloader)))

### Installation
```bash
git clone https://github.com/your-org/comtrade-conversion-weights.git
cd comtrade-conversion-weights
poetry install && poetry shell
cd generator
module load matlab

# Load Comtrade API key
export COMTRADE_API_KEY="your_key_here"
```

## Quick Start

1. **Configure** what conversions you want in `user_config.py`
2. **Run** `python main.py`
3. **Find results** in `data/output/optimized_conversion_weights/`

## Configuration

Edit `user_config.py`:

### Select Adjacent Classification Pairs 
```python
# Individual conversions
CONVERT_HS96_TO_HS92 = True
CONVERT_HS12_TO_HS07 = True
CONVERT_SITC2_TO_SITC1 = False

# Bulk options
ENABLE_ALL_FORWARD = False     # All forward conversions
ENABLE_ALL_BACKWARD = False    # All backward conversions
ENABLE_ALL_CONVERSIONS = False # Everything
```

### Set Data for Previously Downloaded Comtrade data
```python
RAW_DOWNLOADED_COMTRADE_DATA_PATH = "/path/to/comtrade/data"
```

## Supported Conversions

**Forward (older → newer):**
- SITC1 → SITC2 → SITC3 → HS1992
- HS1992 → HS1996 → HS2002 → HS2007 → HS2012 → HS2017 → HS2022

**Backward (newer → older):**
- HS2022 → HS2017 → HS2012 → HS2007 → HS2002 → HS1996 → HS1992
- HS1992 → SITC3 → SITC2 → SITC1

## Output

Conversion weights saved as:
```
data/output/optimized_conversion_weights/
├── conversion_weights_HS1996_HS1992.csv
├── conversion_weights_HS2012_HS2007.csv
└── ...
```

Each file contains: `source_code, target_code, weight, group_id`


## How It Works

1. **Product Grouping**: Uses UN correspondence tables to identify products linked across classifications
2. **Matrix Building**: Creates trade matrices for source/target classifications and conversion relationships
3. **Optimization**: Solves constrained least squares problem for each product group:
   ```
   minimize: ||Y - XB||²
   subject to: Σ(weights) = 1 for each source product
   ```
4. **Validation**: Ensures weights sum to 1 and meet quality standards

## Repository Structure

```
generator/
├── main.py                          # Main entry point
├── user_config.py                   # Configuration
├── src/
│   ├── config/                      
│   ├── python_objects/              # Core processing classes
│   ├── matlab/                      # MATLAB optimization routines
│   ├── R_code/                      # Product grouping scripts
│   └── scripts/                     # Execution scripts
├── data/
│   ├── static/                      # UN correspondence tables
│   ├── correlation_groups/          # Product group assignments
│   ├── matrices/                    # Optimization inputs
│   └── output/                      # Final weights
└── tests/                           # Validation tests
```

## Data Requirements

Expects raw Comtrade data in this structure:
```
/path/to/comtrade/data/as_reported/raw_parquet/
├── H0/1992/COMTRADE-*.parquet
├── H4/2012/COMTRADE-*.parquet
├── S2/1976/COMTRADE-*.parquet
└── ...
```

## License

Apache License, Version 2.0 - see LICENSE file.

## Citation
[![DOI](https://zenodo.org/badge/987233396.svg)](https://doi.org/10.5281/zenodo.16051508)

```bibtex
@Misc{comtrade_conversion_weights,
  author={Harvard Growth Lab}
  title={Comtrade Conversion Weights Generator},
  year={2025},
  howpublished = {\url{https://github.com/your-org/comtrade-conversion-weights}},
}
```
