# Multi-Omics Integrative Analysis of Neuroinflammation in Psychiatric Disorders

## Project Overview
This repository contains computational pipelines for integrative multi-omics analysis of immune and metabolic dysregulation in psychiatric and neurodevelopmental disorders. Our study integrates multi-tissue transcriptomic data, genetic pleiotropy analysis, and functional validation to elucidate molecular mechanisms underlying neuroinflammation.

## Analysis Workflow
## Python Scripts

​**DLmodels.py**​
A Python script dedicated to defining, training, or deploying deep learning (DL) models. It may include neural network architectures, pre - trained model loading, training pipelines, and prediction workflows.

## R Scripts

​**module_gene_plot.R**：​ 
This R script focuses on visualizing gene modules. It can handle gene - module related data (such as module membership, module eigengenes) and generate plots (e.g., using ggplot2) to illustrate module characteristics clearly.

​**mergeomic_network.R**： ​
Used for integrating multi - omics data (e.g., genomics, transcriptomics, proteomics) and constructing networks. It might process cross - omics associations, build network structures (like co - expression or regulatory networks), and analyze interactions across different omics layers.

​**volcano_plot.R**： ​
Specialized in creating volcano plots, a key visualization in differential expression analysis. It takes inputs like fold changes and p - values of genes and generates plots to highlight significantly upregulated or downregulated genes.

​**network_analysis.R**： ​
Performs comprehensive network analysis. It can compute network topological features (e.g., degree centrality, betweenness centrality), detect communities, and explore relationships within biological or other complex networks (e.g., gene regulatory, protein - protein interaction networks).

​**go_heatmap_plotting.R**： ​
Generates heatmaps based on Gene Ontology (GO) annotations. It visualizes the enrichment or expression patterns of genes across different GO functional categories, helping to interpret the functional roles of genes.

​**Gene_shared_specific_analysis.R**： ​
Analyzes the shared and specific gene features across different samples, tissues, or experimental conditions. For example, it can compare gene expression profiles to identify genes that are commonly expressed in some groups or uniquely expressed in a particular subset.

​**Forestplot.R**： ​
Creates forest plots, which are commonly used in meta - analyses to display effect sizes and confidence intervals. It aggregates results from multiple studies or experiments for easy comparison and interpretation.

​**bipartite_network.R**： ​
Focuses on bipartite networks, where nodes are divided into two disjoint sets and edges connect nodes from different sets only. It can be applied to model relationships like gene - disease associations, metabolite - reaction networks, etc., and perform relevant analyses.

​**MR_analysis.R**： ​
Implements Mendelian Randomization (MR) analysis. MR uses genetic variants as instrumental variables to infer causal relationships between exposures and outcomes. This script covers workflows from data preprocessing to causal effect estimation.

​**process_vip.R**： ​
Handles Variable Importance in Projection (VIP) values, which are used in methods like Partial Least Squares Regression (PLSR) to assess variable importance. The script may perform tasks such as VIP value sorting, selecting important variables, and further analysis based on VIP metrics.
