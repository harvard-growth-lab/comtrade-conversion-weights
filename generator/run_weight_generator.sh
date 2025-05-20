#!/bin/bash

# Load required modules
module load R

# Configuration
# =============
# Define the iterations to process
# Format: from_year,to_year,source_classification,target_classification
# Example: 1962,1976,S1,S2 means convert from S1 (1962) to S2 (1976)
ITERATIONS=(
    # SITC (S) conversions
    # S1 <-> S2 conversions
    "1962,1976,S1,S2"  # Convert from S1 (1962) to S2 (1976)
    "1976,1962,S2,S1"  # Convert from S2 (1976) to S1 (1962)
    
    # S2 <-> S3 conversions
    "1988,1976,S3,S2"  # Convert from S3 (1988) to S2 (1976)
    "1976,1988,S2,S3"  # Convert from S2 (1976) to S3 (1988)
    
    # S3 conversions
    "1988,1992,S3,H0"  # Convert from S3 (1988) to H0 (1992)
    
    # HS (H) conversions
    # H0 conversions
    "1992,1988,H0,S3"  # Convert from H0 (1992) to S3 (1988)
    
    # Additional HS conversions (previously commented)
    # H0 <-> H1 conversions
    "1996,1992,H1,H0"  # Convert from H1 (1996) to H0 (1992)
    "1992,1996,H0,H1"  # Convert from H0 (1992) to H1 (1996)
    
    # H1 <-> H2 conversions
    "2002,1996,H2,H1"  # Convert from H2 (2002) to H1 (1996)
    "1996,2002,H1,H2"  # Convert from H1 (1996) to H2 (2002)
    
    # H2 <-> H3 conversions
    "2007,2002,H3,H2"  # Convert from H3 (2007) to H2 (2002)
    "2002,2007,H2,H3"  # Convert from H2 (2002) to H3 (2007)
    
    # H3 <-> H4 conversions
    "2012,2007,H4,H3"  # Convert from H4 (2012) to H3 (2007)
    "2007,2012,H3,H4"  # Convert from H3 (2007) to H4 (2012)
    
    # H4 <-> H5 conversions
    "2017,2012,H5,H4"  # Convert from H5 (2017) to H4 (2012)
    "2012,2017,H4,H5"  # Convert from H4 (2012) to H5 (2017)
    
    # H5 <-> H6 conversions
    "2022,2017,H6,H5"  # Convert from H6 (2022) to H5 (2017)
    "2017,2022,H5,H6"  # Convert from H5 (2017) to H6 (2022)
)

# Script setup
# ===========
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Create necessary directories if they don't exist
mkdir -p data/temp data/static data/concordance_groups

# Create iterations CSV file
# ========================
echo "from_year,to_year,source_classification,target_classification" > "$SCRIPT_DIR/data/temp/iterations.csv"
for iteration in "${ITERATIONS[@]}"; do
    echo "$iteration" >> "$SCRIPT_DIR/data/temp/iterations.csv"
done

# Run the R script
# ===============
Rscript "$SCRIPT_DIR/src/create_product_groups.R" "$SCRIPT_DIR/data/temp/iterations.csv" "$SCRIPT_DIR"

# Clean up
# ========
# Uncomment the following line if you want to remove the temporary file
#rm "$SCRIPT_DIR/data/temp/iterations.csv"