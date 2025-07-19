### META_ANALYSIS_FORESTPLOT.R ###
# Automated forest plot generator for multi-omics mendelian randomization results
# Generates publication-quality visualizations with dynamic formatting

# --------------------------
# 1. LIBRARY LOADING & SETUP
# --------------------------
# Install missing dependencies automatically
required_packages <- c("forestplot", "dplyr", "grid")
invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}))

# --------------------------
# 2. DATA PREPROCESSING
# --------------------------
# Truncation function for long exposure names (max 35 characters)
truncate_exposure <- function(exposure_name, max_length = 35) {
  ifelse(nchar(exposure_name) > max_length, 
         paste0(substr(exposure_name, 1, max_length - 3), "..."), 
         exposure_name)
}

# Filter top 10 exposures by IVW p-value (smallest p-values first)
ivw_filtered_data <- mrTab %>% 
  dplyr::filter(method == "Inverse variance weighted") %>% 
  dplyr::arrange(pval) %>% 
  dplyr::slice_head(n = 10)

# Validation check for data availability
if (nrow(ivw_filtered_data) == 0) {
  stop("Error: No data available for Inverse Variance Weighted method.", call. = FALSE)
}

# --------------------------
# 3. DYNAMIC PLOTTING PARAMETERS
# --------------------------
# Calculate percentile-based confidence interval limits for visualization bounds
ci_lower_bound <- quantile(as.numeric(mrTab$or_lci95), 0.05, na.rm = TRUE)
ci_upper_bound <- quantile(as.numeric(mrTab$or_uci95), 0.95, na.rm = TRUE)

# Extract unique exposures for plotting (top 10 by p-value)
target_exposures <- unique(ivw_filtered_data$exposure)

# Initialize data storage structures
plot_data <- list()
method_counts <- numeric()  # Track methods per exposure for line positioning

# --------------------------
# 4. TABLE TEXT CONSTRUCTION
# --------------------------
# Main table construction loop
for (current_exposure in target_exposures) {
  # Subset data for current exposure
  exposure_subset <- mrTab %>% 
    dplyr::filter(exposure == current_exposure)
  
  # Apply name truncation
  formatted_name <- truncate_exposure(current_exposure)
  
  # Extract statistical metrics with proper type conversion
  analysis_methods <- as.character(exposure_subset$method)
  odds_ratios <- as.numeric(exposure_subset$or)
  ci_lower <- as.numeric(exposure_subset$or_lci95)
  ci_upper <- as.numeric(exposure_subset$or_uci95)
  p_values <- sprintf("%.2e", exposure_subset$pval)
  
  # Apply dynamic CI bounds
  ci_lower <- pmax(ci_lower, ci_lower_bound)
  ci_upper <- pmin(ci_upper, ci_upper_bound)
  
  # Count methods for this exposure (for line positioning)
  method_counts <- c(method_counts, length(analysis_methods))
  
  # Construct formatted table text
  exposure_tabletext <- cbind(
    c(formatted_name, analysis_methods),  # Exposure name + methods
    c("", sprintf("%.2f", odds_ratios)),  # OR values (2 decimal places)
    c("", paste(sprintf("%.3g", ci_lower), "-", sprintf("%.3g", ci_upper))),  # CI (3 sig figs)
    c("", p_values)  # P-values (scientific notation)
  )
  
  # Append to master table
  plot_data <- rbind(plot_data, exposure_tabletext)
  
  # Prepare numerical vectors for forestplot
  mean_values <- c(mean_values, NA, odds_ratios)
  lower_bounds <- c(lower_bounds, NA, ci_lower)
  upper_bounds <- c(upper_bounds, NA, ci_upper)
  
  # Mark summary rows (exposure headers)
  is_summary_rows <- c(is_summary_rows, TRUE, rep(FALSE, length(analysis_methods)))
}

# --------------------------
# 5. PLOT CONFIGURATION
# --------------------------
# Add column headers to table text
column_headers <- c("Exposure", "Odds Ratio", "95% CI", "P-value")
plot_data <- rbind(column_headers, plot_data)

# Initialize numerical vectors with header row
mean_values <- c(NA, mean_values)
lower_bounds <- c(NA, lower_bounds)
upper_bounds <- c(NA, upper_bounds)
is_summary_rows <- c(TRUE, is_summary_rows)

# Define visual styling parameters
text_colors <- fpColors(
  box = "#0072B2",       # Deep blue for effect size boxes
  line = "#D55E00",      # Orange-red for confidence intervals
  summary = "#009E73",   # Teal for summary rows
  zero = "#CC79A7"       # Magenta for reference line
)

# Horizontal rule configuration
horizontal_rules <- list(
  "1" = gpar(lwd = 0),  # No rule before header
  "2" = gpar(lwd = 2.5, col = "#2F4F4F")  # Bold divider after header
)

# Dynamic rule positions after each exposure block
current_row_position <- 2  # Start after header
for (method_count in method_counts) {
  current_row_position <- current_row_position + 1 + method_count
  horizontal_rules[[as.character(current_row_position)]] <- gpar(lwd = 1, col = "#E6E6E6")
}

# --------------------------
# 6. PLOT GENERATION
# --------------------------
# Configure output file (anonymous path)
output_filename <- paste0("mr_results_forestplot_", Sys.Date(), ".pdf")
pdf(output_filename, width = 14, height = 16)

# Render final forest plot
forestplot(
  labeltext = plot_data,
  mean = mean_values,
  lower = lower_bounds,
  upper = upper_bounds,
  title = "Top 10 Exposures: IVW Method Results",
  xlab = "Odds Ratio (95% Confidence Interval)",
  zero = 1,  # Reference line at null effect
  boxsize = 0.12,
  col = text_colors,
  xticks = seq(
    max(ci_lower_bound, round(min(lower_bounds, na.rm = TRUE), 1)),
    min(ci_upper_bound, round(max(upper_bounds, na.rm = TRUE), 1)),
    by = 0.2
  ),
  is.summary = is_summary_rows,
  txt_gp = fpTxtGp(
    label = gpar(cex = 0.85, fontfamily = "Helvetica", col = "#333333"),
    xlab = gpar(cex = 1.0, fontfamily = "Helvetica", fontface = "italic"),
    title = gpar(cex = 1.2, fontfamily = "Helvetica", fontface = "bold", col = "#1F2A44"),
    ticks = gpar(cex = 0.9, fontfamily = "Helvetica")
  ),
  align = c("l", "c", "c", "c"),
  hrzl_lines = horizontal_rules,
  lineheight = unit(0.8, "cm"),
  grid = TRUE,
  colgrid = gpar(lwd = 0.5, col = "#F0F0F0"),
  lwd.zero = 3,
  lwd.ci = 1.5,
  ci.vertices = TRUE,
  ci.vertices.height = 0.1,
  clip = c(
    max(ci_lower_bound - 0.15, min(lower_bounds, na.rm = TRUE) - 0.15),
    min(ci_upper_bound + 0.15, max(upper_bounds, na.rm = TRUE) + 0.15)
  ),
  graph.pos = 2,
  background.args = list(fill = c("#F9FAFB", "#FFFFFF"))
)

# Close graphics device
dev.off()

# --------------------------
# 7. COMPLETION MESSAGE
# --------------------------
message(sprintf(
  "Forest plot successfully generated: %s
",
normalizePath(output_filename, winslash = "/", mustWork = FALSE)
))