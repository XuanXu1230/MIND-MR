# Load necessary libraries
library(TwoSampleMR)
library(ieugwasr)
library(dplyr)
library(openxlsx)
library(extrafont)

# Set working directory (update with your path)
setwd("~/mr/")

# 1. Define file paths and variables (no need to explicitly define specific outcomes)
exposureFile <- "exposure.F.csv"            # Exposure data file

# Load exposure data
exposure_dat <- read_exposure_data(filename = exposureFile,
                                   sep = ",",
                                   snp_col = "SNP",
                                   beta_col = "beta.exposure",
                                   se_col = "se.exposure",
                                   pval_col = "pval.exposure",
                                   effect_allele_col = "effect_allele.exposure",
                                   other_allele_col = "other_allele.exposure",
                                   eaf_col = "eaf.exposure",
                                   phenotype_col = "exposure",
                                   id_col = "id.exposure",
                                   samplesize_col = "samplesize.exposure",
                                   chr_col = "chr.exposure", 
                                   pos_col = "pos.exposure",
                                   clump = FALSE)

# 2. Prepare the exposure data (rename columns)
data1 <- dplyr::select(exposure_dat, SNP, chr.exposure, pos.exposure)
colnames(data1) <- c("rsid", "chromosome", "position")

# 3. Find the nearest genes for the exposure data SNPs
result <- find_nearest_gene(data1, flanking = 50, build = "hg19", collapse = TRUE, snp = "rsid", chr = "chromosome", bp = "position")

# 4. Match SNPs to genes from the exposure data
matched_snp <- result %>%
  filter(sapply(GENES, function(x) any(grepl(paste(sorted_result$Gene, collapse = "|"), x)))) %>%
  dplyr::select(rsid, chromosome, position, GENES) %>%
  mutate(matched_genes = sapply(GENES, function(x) {
    matched <- intersect(strsplit(x, ",")[[1]], sorted_result$Gene)
    paste(matched, collapse = ", ")
  })) %>%
  filter(matched_genes != "")

# 5. Display matched SNPs
print(matched_snp)

# 6. Update exposure data with matched SNPs
match_id <- exposure_dat[exposure_dat$SNP %in% matched_snp$rsid,]$id.exposure
exposure_dat <- exposure_dat[exposure_dat$id.exposure %in% match_id,]

# 7. Extract outcome data based on SNPs
outcomeID <- "outcome_data_id"  # Placeholder for outcome ID
outcomeData <- extract_outcome_data(snps = exposure_dat$SNP, outcomes = outcomeID)

# 8. Process SNP batches for large datasets
batch_size <- 200
total_snps <- length(exposure_dat$SNP)
batches <- ceiling(total_snps / batch_size)

# 9. Function to standardize column order
standardize_columns <- function(data, reference_columns) {
  missing_cols <- setdiff(reference_columns, colnames(data))
  if (length(missing_cols) > 0) {
    data[missing_cols] <- NA
  }
  data <- data[, reference_columns, drop = FALSE]
  return(data)
}

# 10. Initialize empty lists to store results
results_list <- list()
failed_batches <- list()

# 11. Function to extract data with retry mechanism
extract_with_retry <- function(snp_batch, outcomes, retries = 3) {
  attempt <- 1
  while (attempt <= retries) {
    tryCatch({
      cat("Attempting extraction (try", attempt, "of", retries, ")\n")
      return(extract_outcome_data(snps = snp_batch, outcomes = outcomes))
    }, error = function(e) {
      cat("Error occurred on attempt", attempt, ":", e$message, "\n")
    })
    attempt <- attempt + 1
  }
  return(NULL)
}

# 12. Loop through SNP batches to extract outcome data
reference_columns <- NULL
for (i in 1:batches) {
  start_index <- ((i - 1) * batch_size) + 1
  end_index <- min(i * batch_size, total_snps)
  snp_batch <- exposure_dat$SNP[start_index:end_index]
  
  cat("Extracting data for SNPs", start_index, "to", end_index, "...\n")
  current_data <- extract_with_retry(snp_batch, outcomeID)
  
  if (!is.null(current_data)) {
    if (is.null(reference_columns)) {
      reference_columns <- colnames(current_data)
    }
    current_data <- standardize_columns(current_data, reference_columns)
    results_list[[i]] <- current_data
  } else {
    failed_batches <- c(failed_batches, i)
  }
  
  cat("Completed batch", i, "out of", batches, "\n")
}

# 13. Combine all results into one data frame
combined_results <- do.call(rbind, results_list)
cat("Merging completed. Total SNPs processed:", total_snps, "\n")

# 14. Retry failed batches if any
if (length(failed_batches) > 0) {
  cat("Retrying failed batches...\n")
  
  for (i in failed_batches) {
    snp_batch <- exposure_dat$SNP[((i - 1) * batch_size) + 1:min(i * batch_size, total_snps)]
    current_data <- extract_with_retry(snp_batch, outcomeID)
    if (!is.null(current_data)) {
      current_data <- standardize_columns(current_data, reference_columns)
      results_list[[i]] <- current_data
    }
  }
  
  combined_results <- do.call(rbind, results_list)
  cat("Retry completed. Total SNPs processed:", total_snps, "\n")
}

# 15. Save the combined outcome data
outcomeData <- combined_results
write.csv(outcomeData, file = paste0("outcome-meta.csv"), row.names = FALSE)
save(outcomeData, file = paste0("outcome-meta.Rdata"))

# 16. Harmonize exposure and outcome data
outcomeData$outcome <- "disease_name"  # Placeholder for disease name
dat <- harmonise_data(exposure_dat, outcomeData)

# 17. Filter for valid data (mr_keep == TRUE)
outTab <- dat[dat$mr_keep == "TRUE",]
write.csv(outTab, file = paste0("table.SNP.csv"), row.names = FALSE)

# 18. Perform Mendelian randomization analysis
mrResult <- mr(dat)

# 19. Filter significant results for 'Inverse variance weighted' method with p-value < 0.05
mrResult_sig <- mrResult %>%
  filter(method == "Inverse variance weighted" & pval < 0.05)

# 20. Extract all methods for significant results
mrResult_sig_all_methods <- mrResult %>%
  filter(id.exposure %in% mrResult_sig$id.exposure & id.outcome %in% mrResult_sig$id.outcome)

# 21. Calculate odds ratios (OR) for the significant results
mrTab <- generate_odds_ratios(mrResult_sig_all_methods)
write.csv(mrTab, file = paste0("MRresult-meta.csv"), row.names = FALSE)

# 22. Create and save a forest plot
pdf(file = paste0("pic.forest_plot.pdf"), width = 10, height = 15)
ggforest_forest_plot(mrTab)
dev.off()

# 23. Merge significant data with MR results
dat_sig <- dplyr::inner_join(dat, mrResult_sig_all_methods[, c('id.outcome', 'id.exposure')], by = c("id.outcome", "id.exposure"))

# 24. Heterogeneity analysis
heterTab <- mr_heterogeneity(dat_sig)
write.csv(heterTab, file = paste0("heterogeneity-meta.csv"), row.names = FALSE)

# 25. Pleiotropy test
pleioTab <- mr_pleiotropy_test(dat_sig)

# 26. Evaluate the MR results, heterogeneity, and pleiotropy
evaluate_mr_results <- function(mrTab, heterTab, pleioTab) {
  if (nrow(mrTab) == 0) {
    return(data.frame(
      Outcome_ID = "No significant results",
      Exposure_ID = "No significant results",
      Significant_Effect = "Not Significant",
      Heterogeneity = "No significant heterogeneity",
      Pleiotropy = "No significant pleiotropy",
      Robustness = "Caution: No significant results"
    ))
  }
  
  pairs <- unique(mrTab[, c("id.outcome", "id.exposure")])
  evaluation <- data.frame(
    Outcome_ID = pairs$id.outcome,
    Exposure_ID = pairs$id.exposure,
    Significant_Effect = NA_character_,
    Heterogeneity = NA_character_,
    Pleiotropy = NA_character_,
    Robustness = NA_character_
  )
  
  for (i in 1:nrow(evaluation)) {
    outcome_id <- evaluation$Outcome_ID[i]
    exposure_id <- evaluation$Exposure_ID[i]
    
    mr_subset <- mrTab[mrTab$id.outcome == outcome_id & mrTab$id.exposure == exposure_id, ]
    heter_subset <- heterTab[heterTab$id.outcome == outcome_id & heterTab$id.exposure == exposure_id, ]
    pleio_subset <- pleioTab[pleioTab$id.outcome == outcome_id & pleioTab$id.exposure == exposure_id, ]
    
    # Evaluate significance, heterogeneity, and pleiotropy
    evaluation$Significant_Effect[i] <- ifelse(any(mr_subset$pval < 0.05, na.rm = TRUE), "Significant", "Not Significant")
    evaluation$Heterogeneity[i] <- ifelse(any(heter_subset$pval < 0.05, na.rm = TRUE), "Significant heterogeneity", "No significant heterogeneity")
    evaluation$Pleiotropy[i] <- ifelse(any(pleio_subset$pval < 0.05, na.rm = TRUE), "Significant pleiotropy", "No significant pleiotropy")
    
    # Check robustness
    evaluation$Robustness[i] <- ifelse(
      evaluation$Significant_Effect[i] == "Significant" & 
        evaluation$Heterogeneity[i] == "No significant heterogeneity" & 
        evaluation$Pleiotropy[i] == "No significant pleiotropy",
      "Robust", "Caution"
    )
  }
  return(evaluation)
}

# 27. Save results as RData and Excel
results_list <- list(
  MR_Results = mrTab,
  Heterogeneity = heterTab,
  Pleiotropy = pleioTab,
  Evaluation_Summary = evaluate_mr_results(mrTab, heterTab, pleioTab)
)

# Save results
save(results_list, file = paste0("MR_Summary-meta.RData"))
saveWorkbook(wb, file = paste0("MR_Summary-meta.xlsx"), overwrite = TRUE)

# Output confirmation
cat("Results saved successfully.\n")
