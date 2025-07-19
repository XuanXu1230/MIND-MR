# Script: process_data.R
# Purpose: To process data, calculate composite scores, and filter important features and models

# 1. Load necessary libraries
library(dplyr)

# 2. Define function to process data
process_data <- function(vip_path, per_path, weight_importance = 0.9, weight_occurrences = 0.1) {
  
  # 2.1 Load data
  # Read VIP data (importance scores for features) from the given path
  data_vip <- read.csv(vip_path, header = TRUE)
  
  # Read performance data (model evaluation metrics) from the given path
  data_performance <- read.csv(per_path, header = TRUE)
  
  # 2.2 Process VIP data - Summarize by Feature
  # Group by Feature and calculate mean Importance, number of occurrences, and list of models and IDs
  vip_summary <- data_vip %>%
    dplyr::group_by(Feature) %>%
    dplyr::summarise(
      Mean_Importance = mean(Importance),          # Mean of Importance for each Feature
      Occurrences = dplyr::n(),                     # Count occurrences of Feature
      Models = paste(unique(Model), collapse = ", "), # List unique Models
      IDs = paste(unique(ID), collapse = ", ")     # List unique IDs
    )
  
  # 2.3 Normalize performance metrics to a 0-1 range
  data_normalized <- data_performance %>%
    dplyr::mutate(
      Accuracy_norm = (Accuracy - min(Accuracy, na.rm = TRUE)) / (max(Accuracy, na.rm = TRUE) - min(Accuracy, na.rm = TRUE)),
      Precision_norm = (Precision - min(Precision, na.rm = TRUE)) / (max(Precision, na.rm = TRUE) - min(Precision, na.rm = TRUE)),
      Recall_norm = (Recall - min(Recall, na.rm = TRUE)) / (max(Recall, na.rm = TRUE) - min(Recall, na.rm = TRUE)),
      F1_Score_norm = (F1.Score - min(F1.Score, na.rm = TRUE)) / (max(F1.Score, na.rm = TRUE) - min(F1.Score, na.rm = TRUE)),
      ROC_AUC_norm = (ROC.AUC - min(ROC.AUC, na.rm = TRUE)) / (max(ROC.AUC, na.rm = TRUE) - min(ROC.AUC, na.rm = TRUE))
    )
  
  # 2.4 Calculate composite score using normalized metrics and weights
  # Define weights for the performance metrics
  weights <- c(Accuracy = 0.2, Precision = 0.2, Recall = 0.2, F1.Score = 0.2, ROC.AUC = 0.2)
  
  data_normalized <- data_normalized %>%
    dplyr::mutate(
      avg_score = Accuracy_norm * weights['Accuracy'] +
        Precision_norm * weights['Precision'] +
        Recall_norm * weights['Recall'] +
        F1_Score_norm * weights['F1.Score'] +
        ROC_AUC_norm * weights['ROC.AUC']
    )
  
  # 2.5 Filter top 50% of models based on composite score
  # Calculate the median composite score to use as a threshold
  threshold <- median(data_normalized$avg_score, na.rm = TRUE)
  
  # Filter the models with avg_score greater than the threshold
  filtered_performance <- data_normalized %>%
    dplyr::filter(avg_score > threshold)
  
  # 2.6 Match filtered IDs and Models with the VIP data
  filtered_vip <- data_vip %>%
    dplyr::semi_join(filtered_performance, by = c("ID", "Model"))
  
  # 2.7 Summarize filtered VIP data by Feature
  result <- filtered_vip %>%
    dplyr::group_by(Feature) %>%
    dplyr::summarise(
      Mean_Importance = mean(Importance),
      Occurrences = dplyr::n(),                     # Count occurrences of Feature
      Models = paste(unique(Model), collapse = ", "),
      IDs = paste(unique(ID), collapse = ", ")
    )
  
  # 2.8 Calculate weighted combined score
  result <- result %>%
    dplyr::mutate(
      combined_score = Mean_Importance * weight_importance +
        Occurrences * weight_occurrences
    )
  
  # 2.9 Sort by combined score in descending order
  sorted_result <- result %>%
    dplyr::arrange(desc(combined_score))
  
  # 2.10 Return sorted results
  return(sorted_result)
}

# 3. Define file paths (paths are placeholders)
# Replace these paths with actual file paths as needed
vip_path <- "/path/to/data/consolidated_feature_importance_results.csv"
per_path <- "/path/to/data/consolidated_performance_results.csv"

# 4. Call the process_data function with the file paths
sorted_result <- process_data(
  vip_path = vip_path, 
  per_path = per_path
)

# 5. View the top few rows of the processed result
head(sorted_result)
