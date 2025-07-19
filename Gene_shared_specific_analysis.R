# Visualization pipeline for shared/specific gene-exposure relationships
# Generates chord diagrams and network visualizations with dynamic formatting

# --------------------------
# 1. LIBRARY LOADING & SETUP
# --------------------------
# Install missing dependencies automatically
required_packages <- c("dplyr", "ggplot2", "stringr", "forcats", "purrr", 
                       "circlize", "ggrepel", "ggraph", "igraph", "scales")
invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}))

# --------------------------
# 2. OUTPUT DIRECTORY CONFIGURATION
# --------------------------
# Define anonymized output directory
output_directory <- "Gene_Exposure_Analysis_Results"
if (!dir.exists(output_directory)) {
  dir.create(output_directory)
}
cat("Output files will be saved in:", normalizePath(output_directory), "
")

# --------------------------
# 3. DATA PREPROCESSING
# --------------------------
# Merge gene mapping with significant genes data
merged_dataset <- inner_join(gene_mapping_sub, significant_genes_with_human, 
                             by = c("GENE_LIST" = "human_symbol"))

# Calculate outcome counts per gene
gene_count_summary <- merged_dataset %>% 
  group_by(GENE_LIST) %>% 
  summarise(n_outcomes = n_distinct(outcome), .groups = 'drop')

# Annotate genes as shared/specific
annotated_data <- merged_dataset %>% 
  left_join(gene_count_summary, by = "GENE_LIST") %>% 
  mutate(gene_type = if_else(n_outcomes > 1, "Shared Gene", "Specific Gene"))

#=======================================================================
# PART 1: SHARED GENES ANALYSIS (RADIAL NETWORK DIAGRAM)
#=======================================================================
# --- Step 1: Data Filtering for Shared Genes ---
shared_genes_data <- annotated_data %>% filter(gene_type == "Shared Gene")

# Select top 3 exposures per gene based on MR p-value
filtered_shared_genes <- shared_genes_data %>%
  group_by(GENE_LIST) %>%
  arrange(pval_mr, .by_group = TRUE) %>%
  slice_head(n = 3) %>%
  ungroup()

if (nrow(filtered_shared_genes) > 0) {
  
  # --- Step 2: Generate Summary Table ---
  summary_table <- filtered_shared_genes %>%
    group_by(GENE_LIST) %>%
    summarise(
      shared_with_outcomes = paste(sort(unique(outcome)), collapse = ", "),
      n_outcomes = n_distinct(outcome),
      top_exposures = paste(
        str_trunc(exposure, 40, "right"),
        collapse = ";\n"
      )
    ) %>%
    rename(gene = GENE_LIST) %>%
    arrange(desc(n_outcomes), gene)
  
  # Save summary table
  write.csv(summary_table, 
            file.path(output_directory, "summary_shared_genes_outcomes_exposures.csv"), 
            row.names = FALSE)
  cat("Shared genes summary table saved.
")
  
  # --- Step 3: Prepare Network Data ---
  # Create outcome-gene links
  outcome_gene_links <- filtered_shared_genes %>%
    distinct(outcome, GENE_LIST) %>%
    rename(from = outcome, to = GENE_LIST)
  
  # Create gene-exposure links
  gene_exposure_links <- filtered_shared_genes %>%
    distinct(GENE_LIST, exposure) %>%
    rename(from = GENE_LIST, to = exposure)
  
  # Add virtual center node connecting all outcomes
  virtual_center <- "Disease Center"
  outcome_names <- unique(filtered_shared_genes$outcome)
  center_links <- data.frame(
    from = virtual_center,
    to = outcome_names
  )
  
  # Combine all edges
  complete_edge_list <- bind_rows(center_links, outcome_gene_links, gene_exposure_links) %>%
    na.omit()
  
  # --- Step 4: Configure Node Attributes ---
  all_node_names <- unique(c(complete_edge_list$from, complete_edge_list$to))
  
  node_attributes <- data.frame(name = all_node_names, stringsAsFactors = FALSE) %>%
    mutate(
      type = case_when(
        name == virtual_center ~ "Center",
        name %in% outcome_names ~ "Outcome",
        name %in% unique(filtered_shared_genes$GENE_LIST) ~ "Gene",
        name %in% unique(filtered_shared_genes$exposure) ~ "Exposure",
        TRUE ~ "Unknown"
      ),
      label = if_else(type == "Exposure", str_trunc(name, 35, "right"), name),
      font_face = if_else(type == "Gene", "bold", "plain")
    )
  
  # --- Step 5: Define Color Scheme ---
  dopamine_palette <- c("#f9a620", "#e57474", "#63b7af", "#5386a6", "#f4b393", "#a8a2d1")
  neutral_color <- "#E0DDCF"
  
  # Map outcomes to colors
  outcome_color_map <- setNames(
    rep(dopamine_palette, length.out = length(unique(outcome_names))),
    unique(outcome_names)
  )
  
  # Map exposures to colors based on first associated outcome
  exposure_color_map <- filtered_shared_genes %>%
    group_by(exposure) %>%
    summarise(first_outcome = first(outcome)) %>%
    {setNames(outcome_color_map[.$first_outcome], .$exposure)}
  
  # Apply colors to nodes
  node_attributes <- node_attributes %>%
    mutate(
      color = case_when(
        type == "Outcome" ~ outcome_color_map[name],
        type == "Exposure" & name %in% names(exposure_color_map) ~ exposure_color_map[name],
        type == "Center" ~ "white",
        TRUE ~ neutral_color
      )
    ) %>%
    mutate(color = ifelse(is.na(color), neutral_color, color))
  
  # --- Step 6: Generate Radial Network Plot ---
  network_graph <- graph_from_data_frame(
    d = complete_edge_list,
    vertices = node_attributes,
    directed = TRUE
  )
  
  radial_plot <- ggraph(network_graph, layout = 'tree', root = virtual_center, circular = TRUE) + 
    geom_edge_diagonal(aes(color = node1.color), width = 0.6, alpha = 0.6) +
    geom_node_point(aes(color = color), size = 6) +
    geom_node_text(
      aes(label = label, fontface = font_face),
      size = 3.5,
      hjust = 'outward',
      angle = -((-node_angle(x, y) + 90) %% 180) + 90
    ) +
    scale_edge_color_identity(guide = 'none') +
    scale_color_identity(
      guide = 'legend',
      name = "Node Type",
      labels = c(names(outcome_color_map), "Gene/Shared Exposure"),
      breaks = c(unname(outcome_color_map), neutral_color)
    ) +
    theme_graph(base_family = 'sans') +
    labs(
      title = "Disease-Centric View of Shared Genetic & Exposure Links",
      subtitle = "Outcomes at center, connected to genes and their top exposures"
    ) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    ) +
    coord_fixed(xlim = c(-1.5, 1.5), ylim = c(-1.5, 1.5))
  
  # Save visualization
  ggsave(
    file.path(output_directory, "shared_genes_radial_plot.pdf"),
    plot = radial_plot,
    width = 15,
    height = 15
  )
  
  cat("Radial network plot generated successfully.
")
  
} else {
  message("No shared genes detected - skipping radial plot generation.")
}