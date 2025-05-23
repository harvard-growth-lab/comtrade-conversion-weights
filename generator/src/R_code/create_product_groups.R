################################################################################
## Function to create Product Groups (i.e., identify product directly or indirectly linked by UN correspondence tables for a given HS vintage)
################################################################################



library(ggplot2)
library(readxl)
library(stringr)
library(CVXR)
library(openxlsx)
library(Hmisc)
library(expss)
library(reshape2)
library("R.matlab")
library("matlabr")
library(splitstackshape)

rm(list = ls())

# Function to parse command line arguments
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 2) {
    stop("Two arguments required: 1) CSV file path containing the iterations dataframe, 2) working directory path")
  }
  iterations_file <- args[1]
  working_dir <- args[2]
  
  if (!file.exists(iterations_file)) {
    stop(sprintf("File %s does not exist", iterations_file))
  }
  if (!dir.exists(working_dir)) {
    stop(sprintf("Directory %s does not exist", working_dir))
  }
  
  # Set working directory
  setwd("/n/hausmann_lab/lab/atlas/bustos_yildirim/weights_generator/generator")
  
  return(read.csv(iterations_file))
}

# Get iterations dataframe from command line
iterations_df <- parse_args()

### The 'create.groups' function:
create.groups=function(data ## Specify a two column dataframe with the code correspondences
                       # direct = direction
){
  
  base=unique(data)
  
  first.col=names(base)[1]; second.col=names(base)[2]
  
  data.table::setnames(base,names(base),c('first.col','second.col'))
  base$group=NA
  
  i=1
  
  while(nrow(subset(base, is.na(group)))>0){
    
    new.code=subset(base, is.na(group))$first.col[1]  
    
    matches.first.col=unique(subset(base, first.col %in% new.code)$second.col)
    matches.second.col=unique(subset(base, second.col %in% matches.first.col)$first.col)
    
    base$group[which(base$second.col %in% matches.first.col)]=i
    base$group[which(base$first.col %in% matches.second.col)]=i
    
    prev.first=c('')
    prev.second=c('')
    
    while(!(setequal(prev.first,matches.first.col) & setequal(prev.second,matches.second.col))){
      
      prev.first=matches.first.col
      prev.second=matches.second.col
      
      matches.first.col=unique(subset(base, first.col %in% matches.second.col)$second.col)
      matches.second.col=unique(subset(base, second.col %in% matches.first.col)$first.col)
      
      base$group[which(base$second.col %in% matches.first.col)]=i
      base$group[which(base$first.col %in% matches.second.col)]=i
      
    }
    
    i=i+1
    
  }
  
  data.table::setnames(base,names(base),c(first.col,second.col,'group.id'))
  
  return(base)
  
}

iterations_df <- data.frame(
  from_year = c(
    1996, 2002, 2007, 2012, 2017, 2022, 1992, 1996, 2002, 2007, 2012, 2017, 1962, 1976, 1992, 1988, 1988, 1976),
  to_year = c(
    1992, 1996, 2002, 2007, 2012, 2017, 1996, 2002, 2007, 2012, 2017, 2022, 1976, 1962, 1988, 1976, 1992, 1988),
  source_classification = c(
    "H1", "H2", "H3", "H4", "H5", "H6", "H0", "H1", "H2", "H3", "H4", "H5", "S1", "S2", "H0", "S3", "S3", "S2"),
  target_classification = c(
    "H0", "H1", "H2", "H3", "H4", "H5", "H1", "H2", "H3", "H4", "H5", "H6", "S2", "S1", "S3", "S2", "H0", "S3"),
  stringsAsFactors = FALSE
)


for(i in 1:nrow(iterations_df)) {
  
  # Extract parameters for this iteration
  from_year <- iterations_df$from_year[i]
  to_year <- iterations_df$to_year[i]
  source_classification <- iterations_df$source_classification[i]
  target_classification <- iterations_df$target_classification[i]
  
  if(from_year > to_year){ 
    direction = "backward"
    year_1 = from_year
    year_2 = to_year
    
  } else if (from_year < to_year){
      direction = "forward"
      year_1 = to_year
      year_2 = from_year
  }
  
  ### Load all HS vintage correspondences:
  all.vintages  <- read.csv("data/comtrade_concordance/HS_consolidated_comtrade_concordances.csv")
  ## comtrade always provides concordance tables in later classification to earlier classification
  conversion <- subset(all.vintages, adjustment == sprintf("%s to %s", year_1, year_2))
  
  conversion$code.before = as.numeric(conversion$code.before)
  conversion$code.after = as.numeric(conversion$code.after)
  
  
  first.assignment = merge(conversion, create.groups(subset(conversion, Relationship!="1:1")[,c(1,2)]), by=names(conversion)[1:2], all.x = T)
  if(direction == "backward"){
    test1 = aggregate(Relationship ~ group.id, subset(first.assignment, Relationship!="n:1"), function(x) length(unique(x)))
  } else if(direction == "forward"){
    test1 = aggregate(Relationship ~ group.id, subset(first.assignment, Relationship!="1:n"), function(x) length(unique(x)))
  }
  
  conversion = merge(conversion, create.groups(subset(conversion, Relationship!="1:1" & code.after %in% subset(first.assignment, group.id %in% test1$group.id)[,1])[,c(1,2)]),
                     by=names(conversion)[1:2], all.x = T)
  
  if(direction=="backward")
  {
    names(conversion)[names(conversion) == "code.after"] <- "code.source"
    names(conversion)[names(conversion) == "code.before"] <- "code.target"
  } else if(direction == "forward"){
    names(conversion)[names(conversion) == "code.after"] <- "code.target"
    names(conversion)[names(conversion) == "code.before"] <- "code.source"
    
  }
  
  write.csv(conversion, file = sprintf("data/concordance_groups/from_%s_to_%s.csv", source_classification, target_classification), row.names=FALSE)

  # Optional: Print status
  cat("Completed conversion from", source_classification, "to", target_classification, "\n")
}

