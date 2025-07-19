### mergeomic_network_v1.R ###
# Multi-omics data integration and bipartite network visualization
# Generates gene-source interaction networks with community detection

# --------------------------
# 1. LIBRARY LOADING
# --------------------------
# Install missing dependencies automatically
required_packages <- c("igraph", "ggraph", "dplyr", "tidyr", "RColorBrewer",
                       "tidygraph", "stringr", "readr")

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
disname <- "ASD"
base_path <- "/DATA/data/MRAD/QTLMR/TWAS/mergeomics/module/"
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
    mutate(
      sample_id = seq_len(nrow(.)),
      source = prefix
    )
}

# Apply processing to all files and combine results
long_data_list <- lapply(file_paths, process_file)
long_data <- bind_rows(long_data_list)

# --------------------------
# 4. SOURCE NAME ABBREVIATION
# --------------------------
# Custom function for source name standardization
abbreviate_source <- function(source_name) {
  # Specific tissue replacements (order matters!)
  source_name <- gsub("^Brain_Hippocampus", "Hippo", source_name, ignore.case = TRUE)
  source_name <- gsub("^Brain_Hypothalamus", "Hypoth", source_name, ignore.case = TRUE)
  source_name <- gsub("^Brain_Cortex", "Cortex", source_name, ignore.case = TRUE)
  
  # General replacements
  source_name <- gsub("^Brain_", "", source_name, ignore.case = TRUE)
  source_name <- gsub("^Whole_Blood", "Blood", source_name, ignore.case = TRUE)
  source_name <- gsub("Frontal_Cortex_BA9", "FC (BA9)", source_name, ignore.case = TRUE)
  source_name <- gsub("Cerebellar_Hemisphere", "Cereb. Hemi.", source_name, ignore.case = TRUE)
  source_name <- gsub("Cerebellum", "Cereb", source_name, ignore.case = TRUE)
  
  # Remove module type suffixes
  source_name <- str_replace(source_name, "\\.MEGENA_[0-9]+$", "")
  source_name <- str_replace(source_name, "\\.MOD_[0-9]+$", "")
  source_name <- str_replace(source_name, "\\.SET_[A-Z]+$", "")
  source_name <- str_replace(source_name, "\\.ANY_META$", "")
  source_name <- str_replace(source_name, "\\.TUMOR$", "")
  source_name <- str_replace(source_name, "\\.CANCER$", "")
  source_name <- str_replace(source_name, "\\.CTX[0-9]+$", "")
  source_name <- str_replace(source_name, "\\.UNIT[0-9]+$", "")
  
  return(source_name)
}

# --------------------------
# 5. BIPARTITE NETWORK CONSTRUCTION
# --------------------------
# Create edge list with abbreviated source names
edge_list_bipartite <- long_data %>%
  filter(!is.na(gene) & !is.na(source) & gene != "" & source != "") %>%
  filter(!str_detect(gene, "^GSM[0-9]+$")) %>%  # Exclude GSM IDs
  mutate(source_abbreviated = sapply(source, abbreviate_source)) %>%
  distinct(gene, source_abbreviated) %>%
  rename(from = gene, to = source_abbreviated)

# Validate edge list
if (nrow(edge_list_bipartite) == 0) {
  message("No valid gene-source pairs found after processing.")
} else {
  # Calculate gene degrees and select top connected genes
  full_gene_degree <- edge_list_bipartite %>%
    group_by(from) %>%
    summarise(n_sources = n(), .groups = "drop") %>%
    rename(gene = from) %>%
    arrange(desc(n_sources))
  
  # Dynamic selection of top genes (20% or minimum threshold)
  num_top_genes <- min(
    ceiling(nrow(full_gene_degree) * 0.2),
    max(3, nrow(full_gene_degree))
  )
  top_genes <- full_gene_degree %>%
    slice_head(n = num_top_genes) %>%
    pull(gene)
  
  # Filter edge list for top genes
  edge_list_filtered <- edge_list_bipartite %>%
    filter(from %in% top_genes)
  
  # Create node lists
  genes_in_network <- unique(edge_list_filtered$from)
  sources_in_network <- unique(edge_list_filtered$to)
  
  nodes <- tibble(
    name = c(genes_in_network, sources_in_network),
    type = c(rep("Gene", length(genes_in_network)), rep("Source", length(sources_in_network)))
  ) %>%
    mutate(id = name)
  
  # Calculate node degrees in filtered network
  gene_degrees <- edge_list_filtered %>%
    group_by(from) %>%
    summarise(n_sources = n(), .groups = "drop") %>%
    rename(name = from)
  
  source_degrees <- edge_list_filtered %>%
    group_by(to) %>%
    summarise(n_genes = n(), .groups = "drop") %>%
    rename(name = to)
  
  nodes <- nodes %>%
    left_join(gene_degrees, by = "name") %>%
    left_join(source_degrees, by = "name") %>%
    mutate(
      degree = ifelse(type == "Gene", n_sources, n_genes),
      degree = ifelse(is.na(degree), 1, degree)
    ) %>%
    filter(degree > 0)
  
  # Final network validation
  if (nrow(nodes) < 2 || nrow(edge_list_filtered) == 0) {
    message("Insufficient nodes/edges for network construction.")
  } else {
    # Create tidygraph object
    bipartite_network <- tbl_graph(nodes = nodes, edges = edge_list_filtered, directed = FALSE)
    
    # --------------------------
    # 6. NETWORK VISUALIZATION
    # --------------------------
    # Community detection
    communities <- tryCatch(
      cluster_louvain(bipartite_network),
      error = function(e) {
        message("Community detection failed: ", e$message)
        NULL
      }
    )
    
    # Visualization parameters
    node_border_color <- "grey35"
    type_colors <- c("Gene" = "#BDE0FE", "Source" = "#FFC8DD")
    
    # Determine coloring variable (type or community)
    if (!is.null(communities) && length(unique(communities$membership)) > 1) {
      bipartite_network <- bipartite_network %>%
        activate(nodes) %>%
        mutate(community = as.factor(communities$membership))
      
      color_palette <- dopamine_pastel_palette(length(unique(communities$membership)))
      coloring_variable <- "community"
    } else {
      color_palette <- type_colors
      coloring_variable <- "type"
    }
    
    # Generate plot
    bipartite_plot <- ggraph(bipartite_network, layout = 'fr') +
      geom_edge_fan(alpha = 0.25, color = "grey70", width = 0.3) +
      geom_node_point(
        aes(size = degree, fill = .data[[coloring_variable]]),
        shape = 21, color = node_border_color, stroke = 0.7
      ) +
      scale_fill_manual(values = color_palette, name = ifelse(!is.null(communities), "Community", "Node Type")) +
      scale_size_continuous(range = c(3, 15), name = "Connections (Degree)") +
      geom_node_text(
        aes(label = name, color = .data[[coloring_variable]]),
        repel = TRUE, size = 2.5, fontface = "bold",
        bg.color = adjustcolor("white", alpha.f = 0.5), bg.r = 0.1,
        max.overlaps = Inf, segment.size = 0.2
      ) +
      guides(fill = guide_legend(override.aes = list(size = 6, shape = 21, stroke = 0.7))) +
      labs(
        title = "Gene-Source Network",
        subtitle = "Nodes colored by community (if detected) or type. Edges show gene presence in source.",
        caption = "Node size: number of connections (degree)."
      ) +
      theme_graph(base_family = "sans", base_size = 12)
    
    # Save plot
    pdf_file <- paste0(disname, "_genes_bipartite_network.pdf")
    ggsave(pdf_file, plot = bipartite_plot, width = 14, height = 10)
    message(paste("Network plot saved to:", pdf_file))
    
    # --------------------------
    # 7. DATA EXPORT
    # --------------------------
    # Save node and edge data
    nodes_data <- as_tibble(activate(bipartite_network, nodes))
    edges_data <- as_tibble(activate(bipartite_network, edges))
    
    write_csv(nodes_data, paste0(disname, "_genes_nodes_data.csv"))
    write_csv(edges_data, paste0(disname, "_genes_edges_data.csv"))
    message("Network data saved to CSV files.")
  }
}

# --------------------------
# 8. DOPAMINE PASTEL PALETTE
# --------------------------
# Function to generate disease-specific color palettes
dopamine_pastel_palette <- function(n) {
  # ASD-specific palette (reversed for better contrast)
  colors <- rev(c(
    "#d2fbd4", "#a5dbc2", "#7bbcb0", "#559c9e", "#3a7c89",
    "#235d72", "#123f5a"
  ))
  
  if (n <= length(colors)) {
    return(colors[1:n])
  } else {
    return(colorRampPalette(colors)(n))
  }
}