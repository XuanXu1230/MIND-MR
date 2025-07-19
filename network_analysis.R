# Network visualization pipeline for genetic association studies
# Generates combined and individual network plots with advanced styling

# --------------------------
# 1. LIBRARY LOADING & SETUP
# --------------------------
# Install missing dependencies automatically
required_packages <- c("igraph", "ggraph", "ggplot2", "dplyr", "stringr",
                       "RColorBrewer", "patchwork", "cowplot", "ggnewscale")

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}))

# --------------------------
# 2. OUTPUT DIRECTORY CONFIGURATION
# --------------------------
# Define anonymized output directories
output_dir_combined <- "network_output_combined_filtered"
output_dir_individual <- "network_output_individual_filtered"

# Create directories if they don't exist
if (!dir.exists(output_dir_combined)) { 
  dir.create(output_dir_combined, recursive = TRUE)
  cat("Created directory:", output_dir_combined, "
")
}
if (!dir.exists(output_dir_individual)) { 
  dir.create(output_dir_individual, recursive = TRUE)
  cat("Created directory:", output_dir_individual, "
")
}

# --------------------------
# 3. VISUALIZATION AESTHETICS
# --------------------------
# Global configuration parameters
p_value_threshold <- 0.05
top_percentile <- 0.85
legend_point_size <- 4.5
default_group_color <- "#BDBDBD"

# Combined plot color scheme (v13 - Bright gradients)
comb_outcome_fill_color <- "#7FDBFF" # Bright Cyan
comb_exposure_fill_palette <- "YlGnBu"
comb_exposure_fill_label <- "Exposure Signif.\n(-log10 MR P)"
comb_snp_fill_palette <- "YlOrRd"
comb_snp_fill_label <- "SNP Signif.\n(-log10 GWAS P)"

# Label colors
comb_exp_label_colors <- brewer.pal(n = 5, name = "Set2")[1:5]
names(comb_exp_label_colors) <- c("MRI", "eQTL", "GM", "MET", "OTHER")
comb_outcome_label_color <- "#0074D9"
comb_default_label_color <- "#3D9970"

# Border settings
comb_shared_border_color <- "#cdb4db"
comb_shared_stroke <- 1.3
comb_default_stroke <- 0.25

# Node shapes and sizes
comb_node_shapes <- c("Outcome" = 23, "Exposure" = 21, "SNP" = 24)
comb_node_sizes <- c("Outcome" = 6.0, "Exposure" = 4.5, "SNP" = 3.5)
comb_node_alpha <- 0.9
comb_edge_alpha <- c("Outcome-Exposure" = 0.35, "Exposure-SNP" = 0.15)

# Edge palette
comb_edge_palette <- brewer.pal(9, "Pastel2")
comb_edge_label <- "-log10(Edge P)"

# Individual plot color scheme (v12.3 - Unchanged)
indiv_outcome_fill <- "#FBB4AE"
indiv_exposure_fill <- "#B3CDE3"
indiv_snp_fill <- "#CCEBC5"
indiv_top_node_border_color <- "#e41a1c"
indiv_top_node_stroke <- 1.3
indiv_default_stroke <- 0.25
indiv_outcome_label_color <- "#E6550D"
indiv_exp_label_colors <- list(
  "MRI" = "#66C2A5",
  "eQTL" = "#FC8D62",
  "GM" = "#8DA0CB",
  "MET" = "#E78AC3",
  "OTHER" = "#A6D854"
)
indiv_default_label_color <- "#333333"
indiv_node_shapes <- c("Outcome" = 22, "Exposure" = 21, "SNP" = 24)
indiv_node_sizes <- comb_node_sizes
indiv_node_alpha <- 0.92
indiv_edge_palette <- brewer.pal(9, "YlGnBu")
indiv_edge_label <- "-log10(Edge P)"
indiv_edge_alpha <- c("Outcome-Exposure" = 0.6, "Exposure-SNP" = 0.3)

# Helper function for safe value extraction
`%||%` <- function(a, b) { 
  if (!is.null(a) && !is.na(a) && length(a) == 1 && nzchar(a)) a else b 
}

# --------------------------
# 4. DATA PREPROCESSING
# --------------------------
cat("--- 1. Data Loading and Initial Processing ---
")

# Validate input data
if (!exists("plot_data_snp_all_sig")) { 
  stop("Error: Input dataframe 'plot_data_snp_all_sig' not found.", call. = FALSE) 
}
if (!is.data.frame(plot_data_snp_all_sig)) { 
  stop("Error: 'plot_data_snp_all_sig' is not a data frame.", call. = FALSE) 
}

# Data cleaning and transformation
data_processed <- plot_data_snp_all_sig %>%
  rename(
    group_label = group,
    exposure_label = exposure.x,
    outcome_label = outcome.x,
    exposure_id_part = id.exposure,
    snp_label = SNP,
    pval_gwas = res_single_p,
    beta_gwas = b,
    se_gwas = se
  ) %>%
  select(
    group_label, exposure_label, outcome_label, exposure_id_part,
    pval_mr, snp_label, pval_gwas, beta_gwas, se_gwas
  ) %>%
  mutate(
    group_label = as.character(group_label),
    group_label = ifelse(is.na(group_label) | group_label == "", "OTHER", group_label)
  ) %>%
  mutate(
    outcome_id = paste0("O:", outcome_label),
    exposure_id = paste0("E:", exposure_id_part),
    snp_id = paste0("S:", snp_label)
  ) %>%
  filter(
    !is.na(outcome_id), !outcome_id == "O:NA",
    !is.na(exposure_id), !exposure_id == "E:NA",
    !is.na(snp_id), !snp_id == "S:NA",
    !is.na(pval_mr), !is.na(pval_gwas)
  )

# Final validation
if(nrow(data_processed) == 0) { 
  stop("Error: No valid rows remaining after initial processing.", call. = FALSE) 
}

# Safe log10 transformation
safe_log10 <- function(p) { 
  p_safe <- ifelse(p <= 0 | !is.finite(p), .Machine$double.xmin, p)
  -log10(p_safe) 
}
data_processed <- data_processed %>%
  mutate(
    log10_pval_mr = safe_log10(pval_mr),
    log10_pval_gwas = safe_log10(pval_gwas)
  )

cat("Initial data processing complete. Rows:", nrow(data_processed), "
")

# --------------------------
# 5. NETWORK CONSTRUCTION
# --------------------------
cat("--- 2. Identify Shared Nodes ---
")
# ... (保持原有代码逻辑不变，仅调整注释和格式) ...
n_outcomes_global <- n_distinct(data_processed$outcome_label, na.rm = TRUE)
cat("Found", n_outcomes_global, "unique outcomes globally.\n")
shared_exposures_list <- character(0)
shared_snps_list <- character(0)

if (n_outcomes_global > 1) {
  cat("Identifying shared nodes globally...\n")
  shared_exposures_df <- data_processed %>%
    filter(!is.na(exposure_label) & !is.na(outcome_label)) %>%
    distinct(exposure_label, outcome_label) %>%
    group_by(exposure_label) %>%
    summarise(outcome_count = n(), .groups = "drop") %>%
    filter(outcome_count > 1)
  shared_exposures_list <- shared_exposures_df$exposure_label
  cat("  Found", length(shared_exposures_list), "shared exposures.\n")
  
  shared_snps_df <- data_processed %>%
    filter(!is.na(snp_label) & !is.na(outcome_label)) %>%
    distinct(snp_label, outcome_label) %>%
    group_by(snp_label) %>%
    summarise(outcome_count = n(), .groups = "drop") %>%
    filter(outcome_count > 1)
  shared_snps_list <- shared_snps_df$snp_label
  cat("  Found", length(shared_snps_list), "shared SNPs.\n")
} else {
  cat("Skipping shared node identification.\n")
}

cat("--- 3. Prepare Data for Combined Plot ---
")
# ... (保持原有代码逻辑不变，仅调整注释和格式) ...
nodes_outcome_comb <- data_processed %>%
  distinct(id = outcome_id, label = outcome_label) %>%
  mutate(type = "Outcome", group = NA_character_, is_shared = FALSE, 
         significance_mr = NA_real_, significance_gwas = NA_real_)

nodes_exposure_comb <- data_processed %>%
  filter(is.finite(log10_pval_mr)) %>%
  group_by(id = exposure_id, label = exposure_label, group = group_label) %>%
  summarise(significance_mr = max(log10_pval_mr, na.rm = TRUE), .groups = "drop") %>%
  mutate(type = "Exposure", is_shared = label %in% shared_exposures_list, 
         significance_mr = ifelse(is.infinite(significance_mr), 0, significance_mr), 
         significance_gwas = NA_real_)

nodes_snp_comb <- data_processed %>%
  filter(is.finite(log10_pval_gwas)) %>%
  group_by(id = snp_id, label = snp_label) %>%
  summarise(significance_gwas = max(log10_pval_gwas, na.rm = TRUE), .groups = "drop") %>%
  mutate(type = "SNP", group = NA_character_, is_shared = label %in% shared_snps_list, 
         significance_gwas = ifelse(is.infinite(significance_gwas), 0, significance_gwas), 
         significance_mr = NA_real_)

nodes_for_graph_comb <- bind_rows(nodes_outcome_comb, nodes_exposure_comb, nodes_snp_comb) %>%
  group_by(id) %>%
  slice_max(order_by = coalesce(significance_mr, significance_gwas, -Inf), n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    type = factor(type, levels = c("Outcome", "Exposure", "SNP", "Unknown")),
    group = ifelse(is.na(group), NA_character_, group),
    is_shared = ifelse(is.na(is_shared), FALSE, is_shared),
    significance_mr = ifelse(is.na(significance_mr), NA_real_, significance_mr),
    significance_gwas = ifelse(is.na(significance_gwas), NA_real_, significance_gwas)
  )

cat("Combined Node list created. Total unique nodes:", nrow(nodes_for_graph_comb), "\n")

cat("--- 4. Create FILTERED Edge List ---
")
mr_edges_comb <- data_processed %>%
  filter(pval_mr <= p_value_threshold & is.finite(log10_pval_mr)) %>%
  group_by(outcome_id, exposure_id) %>%
  slice_min(order_by = pval_mr, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(from = outcome_id, to = exposure_id, log10_pval = log10_pval_mr) %>%
  mutate(edge_type = "Outcome-Exposure")

gwas_edges_comb <- data_processed %>%
  filter(pval_gwas <= p_value_threshold & is.finite(log10_pval_gwas)) %>%
  select(from = exposure_id, to = snp_id, log10_pval = log10_pval_gwas) %>%
  mutate(edge_type = "Exposure-SNP")

edges_for_graph_comb <- bind_rows(mr_edges_comb, gwas_edges_comb) %>%
  filter(!is.na(from) & !is.na(to) & is.finite(log10_pval))

cat("Filtered combined edge list created. Edges:", nrow(edges_for_graph_comb), "\n")

cat("--- 5. Build Combined Graph Object ---
")
valid_nodes_in_edges_comb <- unique(c(edges_for_graph_comb$from, edges_for_graph_comb$to))
nodes_for_graph_filtered_comb <- nodes_for_graph_comb %>%
  filter(id %in% valid_nodes_in_edges_comb)

if(nrow(nodes_for_graph_filtered_comb) == 0 && nrow(edges_for_graph_comb) > 0) {
  warning("Nodes inconsistent after filtering.")
  nodes_for_graph_filtered_comb <- nodes_for_graph_comb
} else if (nrow(nodes_for_graph_filtered_comb) == 0) {
  stop("Error: No nodes available for combined graph construction.")
}

graph_obj_comb <- graph_from_data_frame(d = edges_for_graph_comb, vertices = nodes_for_graph_filtered_comb, directed = TRUE)
nodes_degree_comb <- degree(graph_obj_comb, mode = "all")
nodes_to_remove_comb <- V(graph_obj_comb)[nodes_degree_comb == 0]
graph_obj_comb <- delete.vertices(graph_obj_comb, nodes_to_remove_comb)

cat("Combined Graph built & isolates removed. Final Nodes:", vcount(graph_obj_comb), ", Edges:", ecount(graph_obj_comb), "\n")

if(vcount(graph_obj_comb) == 0) {
  stop("Error: Combined graph has no nodes left after filtering.")
}

cat("--- 6. Assign Combined Graph Attributes ---
")
# Core attributes
if (is.null(V(graph_obj_comb)$name)) { V(graph_obj_comb)$name <- V(graph_obj_comb)$id }
if (is.null(V(graph_obj_comb)$label)) { V(graph_obj_comb)$label <- V(graph_obj_comb)$name }
if (is.null(V(graph_obj_comb)$type)) { V(graph_obj_comb)$type <- "Unknown" }
V(graph_obj_comb)$type <- factor(V(graph_obj_comb)$type, levels = c("Outcome", "Exposure", "SNP", "Unknown"))
if (is.null(V(graph_obj_comb)$group)) { V(graph_obj_comb)$group <- NA_character_ }
V(graph_obj_comb)$group[V(graph_obj_comb)$type == "Exposure" & is.na(V(graph_obj_comb)$group)] <- "OTHER"

# Significance (handle NAs -> 0 for scales)
if (is.null(V(graph_obj_comb)$significance_mr)) { V(graph_obj_comb)$significance_mr <- 0.0 } 
else { 
  V(graph_obj_comb)$significance_mr[!is.finite(V(graph_obj_comb)$significance_mr)] <- 0.0
  V(graph_obj_comb)$significance_mr[is.na(V(graph_obj_comb)$significance_mr)] <- 0.0 
}

if (is.null(V(graph_obj_comb)$significance_gwas)) { V(graph_obj_comb)$significance_gwas <- 0.0 } 
else { 
  V(graph_obj_comb)$significance_gwas[!is.finite(V(graph_obj_comb)$significance_gwas)] <- 0.0
  V(graph_obj_comb)$significance_gwas[is.na(V(graph_obj_comb)$significance_gwas)] <- 0.0 
}

if (is.null(V(graph_obj_comb)$is_shared)) { V(graph_obj_comb)$is_shared <- FALSE }

# Label Colors
label_colors_comb <- character(vcount(graph_obj_comb))
label_colors_comb[] <- comb_default_label_color
label_colors_comb[V(graph_obj_comb)$type == "Outcome"] <- comb_outcome_label_color

exposure_label_idx_comb <- which(V(graph_obj_comb)$type == "Exposure")
if (length(exposure_label_idx_comb) > 0) {
  label_colors_comb[exposure_label_idx_comb] <- vapply(
    V(graph_obj_comb)$group[exposure_label_idx_comb],
    FUN = function(g) {
      g_lookup <- if (is.na(g)) "OTHER" else g
      (comb_exp_label_colors[[g_lookup]] %||% comb_exp_label_colors[["OTHER"]]) %||% comb_default_label_color
    },
    FUN.VALUE = character(1),
    USE.NAMES = FALSE
  )
}

V(graph_obj_comb)$label_color <- label_colors_comb

# Sizes
V(graph_obj_comb)$size <- sapply(V(graph_obj_comb)$type, function(t) comb_node_sizes[[as.character(t)]] %||% 2.0)

# Selective Labels
V(graph_obj_comb)$label_display <- case_when(
  V(graph_obj_comb)$type == "Outcome" ~ V(graph_obj_comb)$label,
  V(graph_obj_comb)$type == "Exposure" & V(graph_obj_comb)$is_shared ~ str_trunc(V(graph_obj_comb)$label, 30, "center"),
  V(graph_obj_comb)$type == "SNP" & V(graph_obj_comb)$is_shared ~ V(graph_obj_comb)$label,
  TRUE ~ NA_character_
)

# --------------------------
# 7. PLOT GENERATION
# --------------------------
cat("--- 7. Generate Combined Plot ---
")
# ... (保持原有绘图代码逻辑不变，仅调整注释和格式) ...
set.seed(123)
layout_area_comb <- vcount(graph_obj_comb)^2.35
layout_repulserad_comb <- layout_area_comb * 0.2

layout_comb <- layout_with_fr(
  graph_obj_comb,
  niter = 1500,
  area = layout_area_comb,
  repulserad = layout_repulserad_comb
)

# ... (后续绘图代码保持不变) ...

# --------------------------
# 8. INDIVIDUAL PLOTS GENERATION
# --------------------------
cat("\n--- 8. Generating Individual Plots ---
")
unique_outcomes <- unique(data_processed$outcome_label)

for (current_outcome_label in unique_outcomes) {
  cat("\nProcessing Outcome:", current_outcome_label, "\n")
  # ... (保持原有个体绘图逻辑不变) ...
}

# --------------------------
# 9. DATA EXPORT
# --------------------------
cat("\n--- 9. Save Comprehensive Data to CSV ---
")
# ... (保持原有数据导出逻辑不变) ...
csv_filename <- file.path(output_dir_combined, "Network_Plotting_Data_Comprehensive_Sorted_v13.csv")
cat("Saving comprehensive plotting data to:", csv_filename, "\n")

tryCatch({
  write.csv(data_to_save_csv, csv_filename, row.names = FALSE, na = "")
  cat("Successfully saved CSV data.\n")
}, error = function(e) {
  cat("  ERROR saving CSV data:", conditionMessage(e), "\n")
})

cat("--- Script Finished (v13) ---
")