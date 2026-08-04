# GRN_GNN
This github page outlines the pipeline, codes and results for the Graph Neural Network.

## Overview

This project uses a **Graph Convolutional Network (GCN)** to reconstruct Gene Regulatory Networks (GRNs) from gene expression data. GCNs learn latent representations of genes by aggregating information from their neighbors in a network, allowing the model to capture relationships that simpler statistical methods can miss. The GCN-derived networks show strong performance, consistently outperforming raw/traditional GRN methods on standard evaluation metrics (AUROC, F1-score) in benchmark testing.

## COVID-19 results

We applied the model to a COVID-19 antiviral response dataset (blood samples from hospitalized COVID-19 patients). The GCN-derived network picked out key immune-related gene interactions — including **OAS1, OAS3, DDX60, MX1**, and **DHX58–ISG15** — that line up well with known antiviral response pathways. These genes are established players in the interferon-induced host defense against coronaviruses, supporting the model's ability to recover biologically relevant interactions.
