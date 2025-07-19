### module_gene_plot_v1.R ###
# Multi-omics module-based gene expression visualization
# Generates violin plots with significance annotations for top differentially expressed genes

# --------------------------
# 1. LIBRARY LOADING
# --------------------------
# Install missing dependencies automatically
required_packages <- c("ggplot2", "dplyr", "tidyr", "ggrepel")

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}))

# --------------------------
# 2. DATA LOADING & PREPROCESSING
# --------------------------
# Define disease name and file paths
disname <- "SCZ"
base_path <- "/DATA/mergeomics/module/"
file_paths <- list.files(
  paste0(base_path, disname, "/MSEA_", disname, "_Brain_Cortex"),
  full.names = TRUE
)

# Add additional tissue-specific file paths
tissue_paths <- c(
  "Brain_Hippocampus",
  "Brain_Hypothalamus",
  "GO",
  "Whole_Blood"
)

for (tissue in tissue_paths) {
  file_paths <- c(
    file_paths,
    list.files(
      paste0(base_path, disname, "/MSEA_", disname, "_", tissue),
      full.names = TRUE
    )
  )
}

# --------------------------
# 3. DATA TRANSFORMATION FUNCTION
# --------------------------
# Function to process individual files into long format
process_file <- function(file_path) {
  # Read and preprocess data
  df <- read.csv(file_path)
  
  # Extract prefix from filename
  file_name <- basename(file_path)
  prefix <- strsplit(file_name, "\\.csv")[[1]][1]
  prefix <- strsplit(prefix, "-")[[1]][2]
  
  # Identify group column and gene columns
  group_col <- grep("Group", colnames(df), value = TRUE)
  gene_cols <- setdiff(colnames(df), group_col)
  
  # Convert to long format with source annotation
  df %>%
    rename(group = !!group_col) %>%
    pivot_longer(cols = all_of(gene_cols), names_to = "gene", values_to = "expression") %>%
    mutate(source = prefix)
}

# Apply processing to all files and combine results
long_data_list <- lapply(file_paths, process_file)
long_data <- bind_rows(long_data_list)

# --------------------------
# 4. DIFFERENTIAL EXPRESSION ANALYSIS
# --------------------------
de_results <- long_data %>%
  group_by(gene) %>%
  summarise(
    log2fc = mean(expression[group == 1]) - mean(expression[group == 0]),
    p_value = tryCatch(t.test(expression ~ group)$p.value, error = function(e) NA),
    .groups = 'drop'
  ) %>%
  mutate(fdr = p.adjust(p_value, method = "fdr")) %>%
  arrange(fdr)

# --------------------------
# 5. TOP GENE SELECTION
# --------------------------
# Select top significant genes
top_de_genes <- de_results %>%
  filter(fdr < 0.05) %>%
  arrange(fdr) %>%
  slice_head(n = 10) %>%
  pull(gene)

# Filter data for selected genes
plot_df <- long_data %>%
  filter(gene %in% top_de_genes) %>%
  mutate(
    group = factor(group),
    gene = factor(gene, levels = top_de_genes)
  )

# --------------------------
# 6. SUMMARY STATISTICS CALCULATION
# --------------------------
summary_stats_df <- plot_df %>%
  group_by(gene, group) %>%
  summarise(
    mean_expr = mean(expression, na.rm = TRUE),
    se_expr = sd(expression, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    numeric_gene_x = as.numeric(gene),
    nudged_x = ifelse(group == "0", numeric_gene_x - 0.2, numeric_gene_x + 0.2)
  )

# --------------------------
# 7. SIGNIFICANCE ANNOTATION
# --------------------------
annotation_df <- de_results %>%
  filter(gene %in% top_de_genes) %>%
  left_join(summary_stats_df %>% 
              group_by(gene) %>% 
              summarise(max_upper_expr_eb = max(mean_expr + se_expr, na.rm = TRUE), .groups = "drop"),
            by = "gene") %>%
  mutate(
    simple_signif_category = ifelse(fdr < 0.05, "Significant (FDR < 0.05)", "Non-significant (FDR ≥ 0.05)"),
    dot_y_position = ifelse(is.na(max_upper_expr_eb) | !is.finite(max_upper_expr_eb),
                            max(plot_df$expression, na.rm = TRUE) + 1,
                            max_upper_expr_eb + (0.18 * abs(max_upper_expr_eb)) + 0.25),
    bracket_label = ""
  ) %>%
  filter(!is.na(dot_y_position) & is.finite(dot_y_position))

# --------------------------
# 8. COLOR SCHEME MANAGEMENT
# --------------------------
get_color_scheme <- function(scheme_name = "pastel_mint") {
  schemes <- list(
    pastel_blue = list(
      violin_colors = list(normal = "#A3BFFA", disease = "#A7DDE8"),
      gradient_colors = c("#E6EFFF", "#A3BFFA", "#6B7280")
    ),
    pastel_mint = list(
      violin_colors = list(normal = "#159ab7", disease = "#CCF0E6"),
      gradient_colors = c("#F0FFFD", "#B2E1D4", "#6D8E87")
    )
  )
  
  if (!scheme_name %in% names(schemes)) {
    warning(paste("Color scheme '", scheme_name, "' not found. Using default.", sep = ""))
    return(list(
      violin_colors = list(normal = "grey80", disease = "grey60"),
      gradient_colors = RColorBrewer::brewer.pal(5, "Greys")
    ))
  }
  return(schemes[[scheme_name]])
}

# --------------------------
# 9. PLOT GENERATION
# --------------------------
create_gradient_violin_plot <- function(
    plot_df_labeled_data,
    summary_stats_df_data,
    annotation_df_data,
    color_scheme = "pastel_mint"
) {
  # Get color scheme
  colors_list <- get_color_scheme(color_scheme)
  violin_colors <- colors_list$violin_colors
  gradient_colors <- colors_list$gradient_colors
  
  # Prepare annotation data
  source_labels <- annotation_df_data %>%
    filter(!is.na(source) & source != "") %>%
    group_by(gene) %>%
    summarise(
      all_sources = list(sort(unique(source))),
      .groups = "drop"
    ) %>%
    mutate(
      sources_to_display = sapply(all_sources, function(sources_list) {
        abbreviated_sources <- sapply(sources_list, function(x) {
          x <- gsub("^Brain_Hippocampus", "Hippo", x, ignore.case = TRUE)
          x <- gsub("^Brain_Hypothalamus", "Hypoth", x, ignore.case = TRUE)
          x <- gsub("^Whole_Blood", "Blood", x, ignore.case = TRUE)
          x <- gsub("^Brain_Cortex", "Cortex", x, ignore.case = TRUE)
          x <- gsub("Frontal_Cortex_BA9", "FC (BA9)", x, ignore.case = TRUE)
          x <- gsub("Cerebellar_Hemisphere", "Cereb. Hemi.", x, ignore.case = TRUE)
          return(x)
        })
        
        truncated_sources <- sapply(abbreviated_sources, function(s) {
          if (nchar(s) > 25) paste0(substr(s, 1, 25), "..") else s
        })
        
        if (length(truncated_sources) > 3) {
          c(truncated_sources[1:3], paste0("(... ", length(truncated_sources) - 3, " more)"))
        } else {
          truncated_sources
        }
      }),
      source = str_wrap(sources_to_display, width = 25)
    ) %>%
    select(gene, source)
  
  # Generate plot
  p <- ggplot()
  
  # Add violin plots
  if (nrow(plot_df_labeled_data) > 0) {
    p <- p + geom_half_violin(
      data = plot_df_labeled_data,
      aes(x = gene, y = expression, fill = group,
          side = ifelse(group == "0", "l", "r")),
      trim = FALSE, alpha = 0.85, linewidth = 0.15, width = 0.7
    )
  }
  
  # Add color scale
  p <- p + scale_fill_manual(
    name = "Patient Group",
    values = c("0" = violin_colors$normal, "1" = violin_colors$disease),
    guide = guide_legend(order = 1)
  )
  
  # Add error bars and mean points
  if (nrow(summary_stats_df_data) > 0) {
    p <- p + geom_errorbar(
      data = summary_stats_df_data,
      aes(x = nudged_x, ymin = mean_expr - se_expr, ymax = mean_expr + se_expr,
          group = interaction(gene, group)),
      width = 0.08, color = "grey50", linewidth = 0.25
    ) +
      geom_point(
        data = summary_stats_df_data,
        aes(x = nudged_x, y = mean_expr, group = interaction(gene, group)),
        shape = 18, size = 1.8, color = "#333333"
      )
  }
  
  # Add significance points
  if (nrow(annotation_df_data) > 0) {
    p <- p + geom_point(
      data = annotation_df_data,
      aes(x = gene, y = dot_y_position + (0.05 * max(summary_stats_df_data$mean_expr + summary_stats_df_data$se_expr, na.rm = TRUE))),
      color = -log10(annotation_df_data$fdr),
      size = 3.2, shape = 16
    ) +
      scale_color_gradientn(
        name = "Significance (FDR)",
        colors = gradient_colors,
        breaks = c(-log10(0.05), -log10(0.01), -log10(0.001)),
        labels = c("0.05", "0.01", "0.001"),
        guide = guide_colorbar(order = 2)
      )
  }
  
  # Add source labels
  if (nrow(source_labels) > 0) {
    p <- p + geom_text(
      data = source_labels,
      aes(x = gene, y = max(summary_stats_df_data$mean_expr + summary_stats_df_data$se_expr, na.rm = TRUE) * 1.1, 
          label = source),
      size = 2.5, color = "#2D2D2D", vjust = 0, hjust = 0.5, lineheight = 0.85
    )
  }
  
  # Finalize plot aesthetics
  p <- p + scale_x_discrete(name = "Gene", labels = function(x) str_wrap(x, width = 7)) +
    scale_y_continuous(name = "Normalized Expression Level", expand = expansion(mult = c(0.05, 0.35))) +
    labs(title = "Top 10 Genes by Significance") +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 13),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 13, face = "bold"),
      legend.position = "right",
      panel.grid = element_blank()
    )
  
  return(p)
}

# --------------------------
# 10. PLOT OUTPUT
# --------------------------
# Generate and save plot
final_plot <- create_gradient_violin_plot(
  plot_df_labeled_data = plot_df,
  summary_stats_df_data = summary_stats_df,
  annotation_df_data = annotation_df,
  color_scheme = "pastel_mint"
)

pdf(paste0(disname, "-mergemodule-viplin.pdf"), width = 18, height = 5)
print(final_plot)
dev.off()

# --------------------------
# 11. DATA EXPORT
# --------------------------
# Save processed data
write.csv(de_results, paste0(disname, "_de_results.csv"), row.names = FALSE)
write.csv(plot_df, paste0(disname, "_plot_data.csv"), row.names = FALSE)
write.csv(annotation_df, paste0(disname, "_annotation_data.csv"), row.names = FALSE)

cat("Module gene plotting completed successfully.
")