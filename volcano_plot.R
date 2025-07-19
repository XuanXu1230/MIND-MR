# Differential gene expression visualization with enhanced styling
# Generates infographic-style volcano plot with smart labeling

# --------------------------
# 1. LIBRARY LOADING
# --------------------------
# Load required packages with automatic installation
required_packages <- c("ggplot2", "dplyr", "ggrepel", "scales", "patchwork")

invisible(lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}))

# --------------------------
# 2. GLOBAL PARAMETERS
# --------------------------
# Statistical thresholds
logfc_threshold <- 1.0
p_value_threshold <- 0.05

# Visual styling parameters
color_up_deep <- "#D62828"       # Vibrant magenta-red
color_up_light <- "#FFDDC1"      # Soft peach
color_down_deep <- "#0077B6"     # Ocean blue
color_down_light <- "#CAF0F8"    # Sky blue
color_nonsig <- "grey88"         # Near-background grey
color_background <- "#FFFFFF"    # Pure white
color_text_dark <- "#212529"     # Professional dark gray
color_text_light <- "grey60"     # Annotation gray

# Transparency settings
color_region_up <- scales::alpha(color_up_deep, 0.05)
color_region_down <- scales::alpha(color_down_deep, 0.05)

# --------------------------
# 3. DATA PREPARATION
# --------------------------
cat("Preparing volcano plot data...
")

# Convert input to tibble and calculate key metrics
df_plot <- significant_genes_with_human %>%
  as_tibble() %>%
  mutate(
    padj = ifelse(padj == 0, .Machine$double.xmin, padj),
    neg_log10_pvalue = -log10(padj),
    is_significant = ifelse(
      is.na(padj) | is.na(log2FoldChange),
      FALSE,
      (padj < p_value_threshold & abs(log2FoldChange) > logfc_threshold)
    ),
    regulation_type = case_when(
      is_significant & log2FoldChange > 0 ~ "Upregulated",
      is_significant & log2FoldChange < 0 ~ "Downregulated",
      TRUE ~ "Non-significant"
    ),
    regulation_type = factor(
      regulation_type,
      levels = c("Upregulated", "Downregulated", "Non-significant")
    )
  )

# --------------------------
# 4. LABEL SELECTION
# --------------------------
cat("Selecting top genes for labeling...
")

# Identify top significant genes for each regulation type
df_plot_sig <- df_plot %>% filter(is_significant)

labels_up <- df_plot_sig %>%
  filter(regulation_type == "Upregulated") %>%
  arrange(padj, desc(abs(log2FoldChange))) %>%
  head(10)

labels_down <- df_plot_sig %>%
  filter(regulation_type == "Downregulated") %>%
  arrange(padj, desc(abs(log2FoldChange))) %>%
  head(10)

genes_to_label_df <- bind_rows(labels_up, labels_down)

# --------------------------
# 5. AXIS RANGE CALCULATION
# --------------------------
cat("Calculating axis ranges...
")

y_max <- if (nrow(df_plot_sig) > 0) {
  max(df_plot_sig$neg_log10_pvalue, na.rm = TRUE) * 1.25
} else {
  -log10(p_value_threshold) + 10
}

x_max_abs <- max(abs(df_plot$log2FoldChange), na.rm = TRUE) * 1.1
x_lim <- max(x_max_abs, logfc_threshold + 1)

# --------------------------
# 6. PLOT CONSTRUCTION
# --------------------------
cat("Building volcano plot...
")

volcano_plot <- ggplot(df_plot, aes(x = log2FoldChange, y = neg_log10_pvalue)) +
  
  # Background annotation regions
  annotate("rect", 
           xmin = logfc_threshold, xmax = x_lim,
           ymin = -log10(p_value_threshold), ymax = y_max,
           fill = color_region_up, color = NA) +
  annotate("rect",
           xmin = -x_lim, xmax = -logfc_threshold,
           ymin = -log10(p_value_threshold), ymax = y_max,
           fill = color_region_down, color = NA) +
  
  # Non-significant points
  geom_point(
    data = . %>% filter(regulation_type == "Non-significant"),
    color = color_nonsig, size = 1.5, alpha = 0.8
  ) +
  
  # Downregulated genes (with fill gradient)
  geom_point(
    data = . %>% filter(regulation_type == "Downregulated"),
    aes(fill = neg_log10_pvalue, size = neg_log10_pvalue),
    shape = 21, color = scales::alpha(color_down_deep, 0.4), stroke = 0.5
  ) +
  scale_fill_gradient(
    low = color_down_light,
    high = color_down_deep,
    guide = "none"
  ) +
  
  # Upregulated genes (with separate fill scale)
  new_scale_fill() +
  geom_point(
    data = . %>% filter(regulation_type == "Upregulated"),
    aes(fill = neg_log10_pvalue, size = neg_log10_pvalue),
    shape = 21, color = scales::alpha(color_up_deep, 0.4), stroke = 0.5
  ) +
  scale_fill_gradient(
    low = color_up_light,
    high = color_up_deep,
    guide = "none"
  ) +
  
  # Global size scale
  scale_size_continuous(range = c(1.5, 8), guide = "none") +
  
  # Smart labeling system
  ggrepel::geom_text_repel(
    data = labels_up,
    aes(label = mouse_symbol),
    size = 3.5, fontface = "italic", color = color_text_dark,
    xlim = c(logfc_threshold * 1.1, NA), direction = "y",
    hjust = 0, box.padding = 0.4, max.overlaps = Inf,
    min.segment.length = 0.2, segment.color = 'grey70', segment.size = 0.4,
    seed = 101
  ) +
  ggrepel::geom_text_repel(
    data = labels_down,
    aes(label = mouse_symbol),
    size = 3.5, fontface = "italic", color = color_text_dark,
    xlim = c(NA, -logfc_threshold * 1.1), direction = "y",
    hjust = 1, box.padding = 0.4, max.overlaps = Inf,
    min.segment.length = 0.2, segment.color = 'grey70', segment.size = 0.4,
    seed = 102
  ) +
  
  # Axis labels and theme
  labs(
    x = expression("log"[2] * " Fold Change"),
    y = expression("-log"[10] * " (Adjusted P-value)")
  ) +
  coord_cartesian(
    xlim = c(-x_lim, x_lim),
    ylim = c(0, y_max),
    clip = "off"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(color = color_text_dark, linewidth = 0.5),
    axis.ticks = element_line(color = color_text_dark),
    axis.text = element_text(color = color_text_dark, size = 12),
    axis.title = element_text(color = color_text_dark, size = 16, face = "bold"),
    plot.background = element_rect(fill = color_background, color = NA),
    plot.margin = margin(80, 25, 25, 25)
  ) +
  
  # Header annotations
  annotate("text",
           x = -x_lim, y = y_max,
           label = "Differential Gene Expression Profile",
           hjust = 0, vjust = 1, size = 8, fontface = "bold", color = color_text_dark) +
  annotate("text",
           x = -x_lim, y = y_max * 0.92,
           label = "Highlighting the top 10 most significant genes in each regulation category.",
           hjust = 0, vjust = 1, size = 4.5, color = color_text_light) +
  annotate("text",
           x = x_lim, y = y_max,
           label = paste(sum(df_plot$regulation_type == "Upregulated"), "\nUPREGULATED"),
           hjust = 1, vjust = 1, size = 6, fontface = "bold", color = color_up_deep, lineheight = 0.9) +
  annotate("text",
           x = x_lim, y = y_max * 0.8,
           label = paste(sum(df_plot$regulation_type == "Downregulated"), "\nDOWNREGULATED"),
           hjust = 1, vjust = 1, size = 6, fontface = "bold", color = color_down_deep, lineheight = 0.9)

# --------------------------
# 7. OUTPUT GENERATION
# --------------------------
cat("Generating output files...
")

# Save plotting data
write.csv(df_plot, "volcano_plot_full_data_infographic.csv", row.names = FALSE)
write.csv(genes_to_label_df, "volcano_plot_labeled_top10_genes_infographic.csv", row.names = FALSE)
cat("Data files saved successfully.
")

# Save PDF visualization
pdf("volcano_plot_infographic_style.pdf", width = 10, height = 10)
print(volcano_plot)
dev.off()
cat("PDF visualization saved to: volcano_plot_infographic_style.pdf
")

# Display plot in RStudio
print(volcano_plot)

cat("Volcano plot generation completed successfully.
")