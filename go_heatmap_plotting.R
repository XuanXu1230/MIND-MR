# GO enrichment analysis and visualization pipeline
# Generates tile plots showing gene-pathway relationships with advanced styling

# --------------------------
# 1. LIBRARY LOADING & SETUP
# --------------------------
# Install missing dependencies automatically
required_packages <- c("clusterProfiler", "org.Rn.eg.db", "org.Mm.eg.db",
                       "AnnotationDbi", "enrichplot", "ggplot2", "dplyr",
                       "tidyr", "RColorBrewer", "openxlsx", "wesanderson",
                       "ggtext", "glue")

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}))

# --------------------------
# 2. DATA PREPROCESSING
# --------------------------
# Prepare gene list for analysis
significant_genes_info <- significant_genes_with_name %>%
  select(mouse_symbol, padj, log2FoldChange) %>%
  filter(!is.na(mouse_symbol) & !is.na(padj) & !is.na(log2FoldChange)) %>%
  distinct(mouse_symbol, .keep_all = TRUE)

# Extract unique gene IDs
significant_gene_ids <- significant_genes_with_name %>%
  filter(!is.na(gene_id)) %>% pull(gene_id) %>% unique()

# Validate input data
if (length(significant_gene_ids) == 0) {
  stop("Error: No valid gene IDs found in input data.", call. = FALSE)
}

# Map Ensembl IDs to Entrez IDs with error handling
entrez_ids <- tryCatch({
  mapIds(org.Mm.eg.db, 
         keys = significant_gene_ids, 
         column = "ENTREZID", 
         keytype = "ENSEMBL", 
         multiVals = "first")
}, error = function(e) {
  # Fallback mapping using gene symbols if Ensembl mapping fails
  gene_symbols <- significant_genes_with_name %>% 
    filter(!is.na(gene_id)) %>% 
    pull(gene_id) %>% 
    unique()
  if (length(gene_symbols) > 0) {
    mapIds(org.Mm.eg.db, 
           keys = gene_symbols, 
           column = "ENTREZID", 
           keytype = "ENSEMBL", 
           multiVals = "first")
  } else {
    NULL
  }
})

# Validate Entrez ID mapping
if (!is.null(entrez_ids)) {
  entrez_ids_clean <- unique(entrez_ids[!is.na(entrez_ids)])
} else {
  entrez_ids_clean <- character(0)
}

if (length(entrez_ids_clean) == 0) {
  stop("Error: No genes could be mapped to Entrez IDs.", call. = FALSE)
}

cat(sprintf("%d input gene IDs mapped to %d unique Entrez IDs\n",
            length(significant_gene_ids),
            length(entrez_ids_clean)))

# --------------------------
# 3. ENRICHMENT ANALYSIS
# --------------------------
# Initialize results storage
enrichment_results <- list()
kegg_results <- NULL

# Perform GO enrichment (BP, MF, CC) with error handling
go_ontologies <- c("BP", "MF", "CC")
for (ontology in go_ontologies) {
  tryCatch({
    ego_result <- enrichGO(
      gene = entrez_ids_clean,
      OrgDb = org.Mm.eg.db,
      keyType = "ENTREZID",
      ont = ontology,
      pAdjustMethod = "BH",
      pvalueCutoff = 0.05,
      qvalueCutoff = 0.2,
      readable = TRUE
    )
    
    if (!is.null(ego_result) && nrow(as.data.frame(ego_result)) > 0) {
      enrichment_results[[ontology]] <- ego_result
    }
  }, error = function(e) {
    message(sprintf("GO %s enrichment failed: %s", ontology, e$message))
  })
}

# Perform KEGG enrichment with error handling
tryCatch({
  kegg_result <- enrichKEGG(
    gene = entrez_ids_clean,
    organism = "mmu",
    keyType = "ncbi-geneid",
    pAdjustMethod = "BH",
    pvalueCutoff = 0.05,
    qvalueCutoff = 0.2
  )
  
  if (!is.null(kegg_result) && nrow(as.data.frame(kegg_result)) > 0) {
    kegg_results <- setReadable(kegg_result, OrgDb = org.Mm.eg.db, keyType = "ENTREZID")
  }
}, error = function(e) {
  message(sprintf("KEGG enrichment failed: %s", e$message))
})

# --------------------------
# 4. RESULTS EXPORT
# --------------------------
# Create output directory
output_dir <- "enrichment_results"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Save results to Excel
if (length(enrichment_results) > 0 || !is.null(kegg_results)) {
  wb <- createWorkbook()
  
  # Save GO results
  if (length(enrichment_results) > 0) {
    for (ontology in names(enrichment_results)) {
      result_df <- as.data.frame(enrichment_results[[ontology]])
      if (nrow(result_df) > 0) {
        addWorksheet(wb, paste0("GO_", ontology))
        writeData(wb, sheet = paste0("GO_", ontology), x = result_df)
      }
    }
  }
  
  # Save KEGG results
  if (!is.null(kegg_results) && nrow(as.data.frame(kegg_results)) > 0) {
    addWorksheet(wb, "KEGG_Pathways")
    writeData(wb, sheet = "KEGG_Pathways", x = as.data.frame(kegg_results))
  }
  
  # Save workbook
  excel_filename <- file.path(output_dir, "all_enrichment_results.xlsx")
  tryCatch({
    saveWorkbook(wb, excel_filename, overwrite = TRUE)
    cat(sprintf("Enrichment results saved to: %s\n", excel_filename))
  }, error = function(e) {
    message(sprintf("Error saving Excel file: %s", e$message))
  })
} else {
  message("No significant enrichment results to save.")
}

# --------------------------
# 5. VISUALIZATION PREPARATION
# --------------------------
# Configuration parameters
top_n_pathways <- 20
max_genes_to_display <- 30

# Initialize data storage for plotting
plot_data_list <- list()

# Process GO results
for (ontology in names(enrichment_results)) {
  result_df <- as.data.frame(enrichment_results[[ontology]])
  
  if (nrow(result_df) >= 1) {
    top_results <- result_df %>%
      arrange(p.adjust) %>%
      head(top_n_pathways) %>%
      mutate(
        geneID_list = strsplit(as.character(geneID), "/"),
        significance_metric = -log10(p.adjust + 1e-10)
      ) %>%
      select(ID, Description, geneID_list, significance_metric) %>%
      unnest(geneID_list) %>%
      rename(
        pathway_id = ID,
        pathway_name = Description,
        gene_symbol = geneID_list
      ) %>%
      left_join(significant_genes_info, by = c("gene_symbol" = "mouse_symbol")) %>%
      mutate(analysis_source = paste0("GO-", ontology))
    
    plot_data_list[[paste0("GO-", ontology)]] <- top_results
  }
}

# Process KEGG results
if (!is.null(kegg_results) && nrow(as.data.frame(kegg_results)) >= 1) {
  kegg_df <- as.data.frame(kegg_results)
  
  if (nrow(kegg_df) >= 1) {
    top_kegg <- kegg_df %>%
      arrange(p.adjust) %>%
      head(top_n_pathways) %>%
      mutate(
        geneID_list = strsplit(as.character(geneID), "/"),
        significance_metric = -log10(p.adjust + 1e-10)
      ) %>%
      select(ID, Description, geneID_list, significance_metric) %>%
      unnest(geneID_list) %>%
      rename(
        pathway_id = ID,
        pathway_name = Description,
        gene_symbol = geneID_list
      ) %>%
      left_join(significant_genes_info, by = c("gene_symbol" = "mouse_symbol")) %>%
      mutate(analysis_source = "KEGG")
    
    plot_data_list[["KEGG"]] <- top_kegg
  }
}

# --------------------------
# 6. DATA INTEGRATION & FILTERING
# --------------------------
if (length(plot_data_list) > 0) {
  # Combine all results
  combined_data <- bind_rows(plot_data_list) %>%
    filter(
      !is.na(gene_symbol) & 
        !is.na(pathway_name) &
        !is.na(padj) & 
        !is.na(log2FoldChange) & 
        !is.na(significance_metric)
    ) %>%
    mutate(
      fill_metric = -log10(padj + 1e-10),
      regulation_shape = factor(
        ifelse(log2FoldChange > 0, "Upregulated", "Downregulated"),
        levels = c("Upregulated", "Downregulated")
      ),
      analysis_source = factor(analysis_source)
    )
  
  if (nrow(combined_data) > 0) {
    # Identify top genes for visualization
    gene_stats <- combined_data %>%
      group_by(gene_symbol) %>%
      summarise(
        pathway_count = n_distinct(pathway_name),
        min_padj = min(padj, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(desc(pathway_count), min_padj)
    
    selected_genes <- head(gene_stats$gene_symbol, max_genes_to_display)
    message(sprintf("Selected %d genes for visualization: %s",
                    length(selected_genes),
                    paste(selected_genes, collapse = ", ")))
    
    # Filter data for selected genes
    final_plot_data <- combined_data %>%
      filter(gene_symbol %in% selected_genes)
    
    # Define pathway ordering
    pathway_order <- final_plot_data %>%
      distinct(pathway_name, analysis_source, significance_metric) %>%
      arrange(analysis_source, desc(significance_metric)) %>%
      pull(pathway_name)
    
    # Apply final ordering
    final_plot_data <- final_plot_data %>%
      mutate(
        pathway_name = factor(pathway_name, levels = rev(pathway_order)),
        gene_symbol = factor(gene_symbol, levels = selected_genes)
      )
    
    # Create source color palette
    source_colors <- setNames(
      c("#A8C4F0", "#B4E4B4", "#F8C8B8", "#E6B8D1"),
      c("GO-BP", "GO-MF", "GO-CC", "KEGG")
    )
    
    # --------------------------
    # 7. VISUALIZATION
    # --------------------------
    tile_plot <- ggplot(final_plot_data, aes(x = gene_symbol, y = pathway_name)) +
      # Background tiles by analysis source
      geom_tile(
        aes(fill = analysis_source),
        color = "#fdfdfd",
        linewidth = 1.2,
        alpha = 0.6
      ) +
      scale_fill_manual(values = source_colors, name = "Analysis Source") +
      new_scale_fill() +
      
      # Data points with significance metrics
      geom_point(
        aes(fill = fill_metric, size = significance_metric, shape = regulation_shape),
        color = "grey40",
        alpha = 0.92,
        stroke = 0.4
      ) +
      scale_fill_gradientn(
        colors = c("#f0f9ff", "#cce5f6", "#99cfe0", "#66b2cc", "#3288bd"),
        name = expression("-log"[10]*italic("P")["adj"]),
        guide = guide_colorbar(barwidth = 1.5, barheight = 5)
      ) +
      scale_size_continuous(
        range = c(2.2, 7.5),
        name = "Pathway Significance"
      ) +
      scale_shape_manual(
        values = c("Upregulated" = 24, "Downregulated" = 25),
        name = "Gene Regulation"
      ) +
      
      # Labels and theme
      labs(
        title = sprintf("Top %d Pathways for %d Selected Genes",
                        top_n_pathways,
                        max_genes_to_display),
        x = "Genes",
        y = NULL
      ) +
      theme_minimal(base_size = 14) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(size = 11),
        legend.position = "right",
        panel.grid = element_blank()
      )
    
    # --------------------------
    # 8. OUTPUT GENERATION
    # --------------------------
    # Calculate dynamic dimensions
    plot_width <- max(8, max_genes_to_display * 0.8)
    plot_height <- max(6, length(pathway_order) * 0.4)
    
    # Save plot
    ggsave(
      filename = file.path(output_dir, "go_enrichment_tile_plot.pdf"),
      plot = tile_plot,
      width = plot_width,
      height = plot_height,
      dpi = 300,
      device = cairo_pdf
    )
    
    # Save plot data
    write.csv(final_plot_data, 
              file = file.path(output_dir, "go_enrichment_plot_data.csv"),
              row.names = FALSE)
    
    message(sprintf("Visualization saved to: %s",
                    file.path(output_dir, "go_enrichment_tile_plot.pdf")))
  } else {
    message("No valid data available for visualization.")
  }
} else {
  message("No enrichment results available for plotting.")
}