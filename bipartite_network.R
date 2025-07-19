# Load required libraries
library(dplyr)
library(igraph)
library(ggraph)
library(ggplot2)
library(stringr)
library(ggrepel)

# Step 1: Filter results based on 'Inverse variance weighted' method and p-value < 0.05
significant_ivw_results <- merged_data %>%
  filter(method == "Inverse variance weighted" & pval < 0.05)

# Step 2: Create a summary table for exposures with significant outcomes
# Count the distinct significant outcomes and list them per exposure
exposure_summary <- significant_ivw_results %>%
  group_by(exposure) %>%
  summarise(
    significant_outcome_count = n_distinct(outcome),  # Count of unique outcomes
    significant_outcomes_list = toString(sort(unique(outcome))),  # List of unique outcomes per exposure
    .groups = 'drop'  # Drop grouping structure
  ) %>%
  rename(outcome = significant_outcomes_list, count = significant_outcome_count)

# Step 3: Remove rows with NA values in 'exposure' or 'outcome'
significant_ivw_results <- significant_ivw_results %>%
  filter(!is.na(exposure), !is.na(outcome))

# Step 4: Summarize the number of unique outcomes for each exposure
exposure_unique_outcomes <- significant_ivw_results %>%
  group_by(exposure) %>%
  summarise(unique_outcome_count = n_distinct(outcome), .groups = "drop") %>%
  filter(unique_outcome_count >= 1) %>%
  arrange(desc(unique_outcome_count))

# Step 5: Print the exposures that have 2 or more unique outcomes for debugging
cat("Exposures with >= 2 unique outcomes:\n")
print(exposure_unique_outcomes, n = Inf)

# Step 6: Create edge list for bipartite network from exposure-outcome pairs
edge_list <- significant_ivw_results %>%
  select(exposure, outcome) %>%
  distinct() %>%
  filter(exposure %in% exposure_unique_outcomes$exposure)

# Step 7: Check edge list summary for validation
cat("Edge list summary:\n")
print(summary(edge_list))

# Step 8: Create node metadata (exposure vs outcome)
nodes <- tibble(
  name = unique(c(edge_list$exposure, edge_list$outcome)),
  type = case_when(
    name %in% edge_list$outcome ~ "outcome",  # Identify outcome nodes
    name %in% edge_list$exposure ~ "exposure"  # Identify exposure nodes
  )
)

# Step 9: Assign color palettes for exposure and outcome types
# Modify for different themes (MetaMR, eQTLMR, MRIMR, GUTMMR)
outcome_colors <- c(
  "AD" = "#d9a7b0",   # Light Teal
  "BIP" = "#a7cbd9",   # Light Coral
  "SCZ" = "#68a0a6",   # Light Olive
  "MDD" = "#476f95",   # Light Purple
  "ADHD" = "#A66F6F",  # Soft Red
  "ASD" = "#7b6992"    # Light Purple
)

count_colors <- c(
  "1" = "#c7cfb7",
  "2" = "#B3CDE3",    # Light Blue
  "3" = "#8C96C6",    # Mid Blue
  "4-5" = "#8856A7",  # Light Purple
  ">5" = "#810F7C"     # Deep Purple
)

# Step 10: Compute node counts from the edge list
node_counts <- edge_list %>%
  filter(exposure %in% nodes$name[nodes$type == "exposure"]) %>%
  group_by(exposure) %>%
  summarise(total_count = n_distinct(outcome)) %>%
  mutate(count_category = case_when(
    total_count == 1 ~ "1",
    total_count == 2 ~ "2",
    total_count == 3 ~ "3",
    total_count %in% 4:5 ~ "4-5",
    total_count > 5 ~ ">5",
    TRUE ~ NA_character_
  )) %>%
  rename(name = exposure)

# Step 11: Validate node counts
if (any(is.na(node_counts$count_category))) {
  stop("Invalid counts in node_counts: ", paste(node_counts$name[is.na(node_counts$count_category)], collapse = ", "))
}

# Step 12: Clean and validate nodes, adding metadata (color, size, shape)
nodes <- nodes %>%
  left_join(node_counts, by = "name") %>%
  mutate(
    total_count = replace_na(total_count, 2),  # Default to 2 if missing
    count_category = replace_na(count_category, "2"),  # Default to category 2 if missing
    type = case_when(
      type == "outcome" ~ "outcome",
      type == "exposure" ~ "exposure",
      TRUE ~ NA_character_
    ),
    color = case_when(
      type == "outcome" ~ outcome_colors[name],
      type == "exposure" ~ count_colors[as.character(count_category)]
    ),
    size = ifelse(type == "outcome", 10, 7),
    shape = ifelse(type == "outcome", "triangle", "circle"),
    label = str_trunc(name, 30),  # Truncate label text for display
    color_label = case_when(
      type == "outcome" ~ name,
      type == "exposure" ~ count_category
    ),
    label_size = ifelse(type == "outcome", 3, 2.5),
    label_alpha = ifelse(type == "outcome", 0.9, 0.6)
  )

# Step 13: Assign edge colors based on outcome values
edge_list <- edge_list %>%
  mutate(outcome = as.character(outcome)) %>%
  mutate(edge_color = case_when(
    outcome %in% names(outcome_colors) ~ outcome_colors[outcome],
    TRUE ~ NA_character_
  ))

# Step 14: Validate edge colors
if (any(is.na(edge_list$edge_color))) {
  stop("Invalid or missing outcome values in edge_list: ",
       paste(unique(edge_list$outcome[is.na(edge_list$edge_color)]), collapse = ", "))
}

# Step 15: Validate edge vertices
if (!all(c(edge_list$from, edge_list$to) %in% nodes$name)) {
  stop("Edge vertices not found in nodes: ",
       paste(setdiff(c(edge_list$from, edge_list$to), nodes$name), collapse = ", "))
}

# Step 16: Create an igraph object for the bipartite network
graph <- graph_from_data_frame(edge_list, vertices = nodes, directed = FALSE)
V(graph)$type <- nodes$type == "outcome"  # Logical type for outcome vs exposure

# Step 17: Compute bipartite layout for visualization
layout <- layout_as_bipartite(graph, types = V(graph)$type)

# Step 18: Plot the bipartite network graph
set.seed(123)
tryCatch({
  ggraph(graph, layout = layout) +
    geom_edge_arc(
      aes(colour = edge_color),
      alpha = 0.4,
      edge_width = 0.8,
      strength = 0.05,
      show.legend = TRUE
    ) +
    geom_node_point(
      aes(colour = color, size = size, shape = shape),
      stroke = 2,
      alpha = 0.95
    ) +
    geom_label_repel(
      aes(x = x, y = y, label = label, fill = alpha("white", label_alpha)),
      size = nodes$label_size,
      fontface = ifelse(V(graph)$type, "bold", "plain"),
      color = "black",
      label.size = 0.2,
      box.padding = 0.5,
      point.padding = 0.2,
      segment.colour = "grey60",
      segment.size = 0.2,
      direction = "y",
      max.overlaps = 25,
      force = 1,
      show.legend = FALSE
    ) +
    scale_colour_identity(
      name = "Node Type",
      breaks = node_color_mapping,
      labels = node_color_labels,
      guide = guide_legend(order = 1)
    ) +
    scale_edge_colour_identity(
      name = "Edge (Outcome)",
      guide = guide_legend(order = 2)
    ) +
    scale_size_identity(guide = "none") + # For node sizes
    scale_fill_identity() +
    scale_shape_manual(values = c(circle = 16, triangle = 17), guide = "none") +
    theme_void() +
    theme(
      plot.background = element_rect(fill = "#FFFFFF", colour = NA),
      panel.background = element_rect(fill = "#FFFFFF", colour = NA),
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 12)),
      plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 15)),
      legend.position = "bottom",
      legend.title = element_text(face = "bold", size = 11),
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.4, "cm"),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.spacing.y = unit(0.5, "cm"),
      plot.margin = margin(30, 60, 30, 30)
    ) +
    guides(
      edge_colour = guide_legend(override.aes = list(edge_width = 2, alpha = 0.4))
    ) +
    labs(
      title = "Bipartite Network of Exposures and Outcomes",
      subtitle = "Exposures with ≥ 2 Unique Outcomes, Colored by Count and Outcome"
    ) -> plot
  
  # Save the plot as a high-quality PDF
  ggsave(
    "bipartite_network_outcomes.pdf",
    plot = plot,
    width = 15,
    height = 10,
    device = "pdf",
    dpi = 300
  )
}, error = function(e) {
  message("Bipartite layout failed: ", e$message, "\nTrying force-directed layout (kk) instead.")
  
  # Fallback to force-directed layout
  layout_kk <- layout_with_kk(graph, maxiter = 2000, epsilon = 0.01)
  ggraph(graph, layout = layout_kk) +
    geom_edge_arc(
      aes(colour = edge_color),
      alpha = 0.4,
      edge_width = 0.8,
      strength = 0.05,
      show.legend = TRUE
    ) +
    geom_node_point(
      aes(colour = color, size = size, shape = shape),
      stroke = 2,
      alpha = 0.95
    ) +
    geom_label_repel(
      aes(x = x, y = y, label = label, fill = alpha("white", label_alpha)),
      size = nodes$label_size,
      fontface = ifelse(V(graph)$type, "bold", "plain"),
      color = "black",
      label.size = 0.2,
      box.padding = 0.5,
      point.padding = 0.2,
      segment.colour = "grey60",
      segment.size = 0.2,
      direction = "y",
      max.overlaps = 25,
      force = 1,
      show.legend = FALSE
    ) +
    scale_colour_identity(
      name = "Node Type",
      breaks = node_color_mapping,
      labels = node_color_labels,
      guide = guide_legend(order = 1)
    ) +
    scale_edge_colour_identity(
      name = "Edge (Outcome)",
      guide = guide_legend(order = 2)
    ) +
    scale_size_identity(guide = "none") +
    scale_fill_identity() +
    scale_shape_manual(values = c(circle = 16, triangle = 17), guide = "none") +
    theme_void() +
    theme(
      plot.background = element_rect(fill = "#FFFFFF", colour = NA),
      panel.background = element_rect(fill = "#FFFFFF", colour = NA),
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 12)),
      plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 15)),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 11),
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.4, "cm"),
      legend.background = element_rect(fill = "white", colour = NA),
      legend.spacing.y = unit(0.5, "cm"),
      plot.margin = margin(30, 60, 30, 30)
    ) +
    guides(
      edge_colour = guide_legend(override.aes = list(edge_width = 2, alpha = 0.4))
    ) +
    labs(
      title = "Network of Exposures and Outcomes (Force-Directed)"
    ) -> plot
  
  # Save the fallback plot
  ggsave(
    "bipartite_network_outcomes_fallback.pdf",
    plot = plot,
    width = 12,
    height = 10,
    device = "pdf",
    dpi = 300
  )
})

# Final message
cat("Plotting completed. Output saved successfully.\n")
