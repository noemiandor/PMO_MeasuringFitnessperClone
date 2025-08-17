library(dyno)
library(tidyverse)
library(ggplot2)
library(dynwrap) 

# Retrieve command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Initialize an input_file variable - this file should be the features csv file from cell cycle classification CNN
input_file <- NULL 

# 1. If command-line args exist, use the first one
if (length(args) > 0) {
  input_file <- args[1]
}

# 2. If no command-line args but we're in interactive mode, prompt the user.
if (is.null(input_file) && interactive()) {
  input_file <- file.choose()  
}

# 3. If still no input_file, throw an error
if (is.null(input_file)) {
  stop("No input file specified. 
       Usage in non-interactive mode: Rscript my_script.R <path_to_file>")
}


# Read the CSV file
data <- read.csv(input_file, stringsAsFactors = FALSE)

#create barcode from the first two cols
data <- unite(data, col="barcode", c('FoF', 'Cell_id'), sep='-')

#merge identically barcoded cells by averaging (average of all slices)
#grp stats automatically sets barcode as rownames
devtools::source_url("https://github.com/noemiandor/Utils/blob/master/grpstats.R?raw=TRUE")
data_merged <- grpstats(x=as.matrix(data[,-1]), g=data$barcode, statscols='median')$median

#if not merging create the rownames manually
rownames(data) <- make.unique(data$barcode)
data <- data[,-1] 

###USE MERGED data###
data <- data_merged

# Rename the first two columns for clarity
colnames(data)[1:2] <- c("real_cell_cycle_label", "inferred_cell_cycle")

# Extract features (all columns except the first two)
features_numeric <- data[, c(-1,-2)]

# Convert all features to numeric
#features_numeric <- data.frame(lapply(features, function(x) as.numeric(as.character(x))))
#rownames(features_numeric) <- rownames(data)  # Set row names on features_numeric

# Remove columns with all NA values (non-numeric columns)
features_numeric <- features_numeric[, colSums(is.na(features_numeric)) < nrow(features_numeric)]

# Convert to matrix
features_numeric_matrix <- as.matrix(features_numeric)
rownames(features_numeric_matrix) <- rownames(features_numeric)  # Ensure row names are preserved
# Check and remove zero-variance columns
zero_variance_cols <- apply(features_numeric_matrix, 2, function(x) var(x, na.rm = TRUE) == 0)
if (any(zero_variance_cols)) {
  features_numeric_matrix <- features_numeric_matrix[, !zero_variance_cols, drop = FALSE]
  print(paste("Removed", sum(zero_variance_cols), "zero-variance columns."))
}
# Median Normalization
feature_medians <- apply(features_numeric_matrix, 2, median, na.rm = TRUE)

# Replace zero or NA medians to avoid division by zero
feature_medians[feature_medians == 0 | is.na(feature_medians)] <- 1e-6

features_normalized <- sweep(features_numeric_matrix, 2, feature_medians, FUN = "/")
rownames(features_normalized) <- rownames(features_numeric_matrix)  # Ensure row names are preserved

# Handle non-finite values (remove rows with NA or infinite values)
# Keep track of finite rows
finite_rows <- apply(is.finite(features_normalized), 1, all)

# After handling non-finite values
finite_rows <- apply(is.finite(features_normalized), 1, all)

# Filter features_normalized to keep only rows with finite values
features_normalized <- features_normalized[finite_rows, , drop = FALSE]

# Create data_filtered by filtering the original data to match
data_filtered <- as.data.frame(data[finite_rows, ])

# Ensure row names are preserved
rownames(features_normalized) <- rownames(features_numeric_matrix)[finite_rows]
rownames(data_filtered) <- rownames(data)[finite_rows]

if (is.null(rownames(features_normalized))) {
  stop("Row names of features_normalized are NULL.")
}

# Perform PCA 
pca_result <- prcomp(features_normalized, scale. = TRUE) 

# Use the PCA scores as the new features
features_pca <- pca_result$x
rownames(features_pca) <- rownames(features_normalized)

#plot #PCA features/variance
variance_explained <- pca_result$sdev^2
proportion_variance <- variance_explained / sum(variance_explained)
cumulative_proportion <- cumsum(proportion_variance)

plot(1:length(cumulative_proportion), cumulative_proportion, 
     type = "b", # "b" for both points and lines
     xlab = "Number of Principal Components", 
     ylab = "Cumulative Proportion of Variance Explained",
     main = "PCA: Variance Explained vs. Number of Components")

# add line for threshold
abline(h = 0.9, col = "red", lty = 2) # Add a dashed red line at 0.98

# Create a dyno dataset (need to include cell_id)
dataset <- wrap_expression(
  counts = features_pca[,1:10],
  expression = features_pca[,1:10]
)

# Infer trajectory using ti_angle
model <- infer_trajectory(
  dataset,
  method = ti_angle(dimred='umap')
)

# Add pseudotime to the filtered data
data_filtered$pseudotime <- model$pseudotime[rownames(data_filtered)]

# Visualize pseudotime versus real cell cycle label
ggplot(data_filtered, aes(x = factor(real_cell_cycle_label), y = pseudotime)) +
  geom_violin(scale='width') +
  xlab("Real Cell Cycle Label") +
  ylab("Pseudotime") +
  ggtitle("Pseudotime vs Real Cell Cycle Label")

# Visualize pseudotime versus inferred cell cycle
#ggplot(data_filtered, aes(x = factor(inferred_cell_cycle), y = pseudotime)) +
 # geom_boxplot(notch = TRUE) +
  #xlab("Inferred Cell Cycle Label") +
  #ylab("Pseudotime") +
  #ggtitle("Pseudotime vs Inferred Cell Cycle")

# Plot pseudotime on a dimensionality reduction
plot_dimred(model, color_cells = "pseudotime")

# Save the model if needed
save(model, file = "~/Downloads/pseudotime_model.Robj")
