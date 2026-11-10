Bayesian Spatial Count Models for Nigeria Terrorism

Reference implementation and analysis code for the manuscript:
"Zero-Inflated Bayesian Spatial Modelling of Terrorism Fatalities in Nigeria: A Comparative Analysis of Areal and Continuous Spatial Specifications"
Overview

This repository contains the canonical R script used to fit the sequence of Bayesian spatial count models (Null NB, Fixed NB, BYM2, and SPDE), perform the 3-seed stability checks, and conduct the spatial cross-validation comparing the Bayesian models against XGBoost.
Repository Structure

    Ng3Terrorism2.R (or .txt): The complete, reproducible R script for all models and diagnostics.

    data/: Instructions for accessing the dataset (NigeriaTerrorism_df.csv).

Requirements

The code was developed and tested using:

    R version 4.5.3

    INLA version 26.06.07

    Required R packages: spdep, ggplot2, viridis, Metrics, xgboost

Reproducibility

The results reported in the manuscript are based on the canonical run using set.seed(2024). Because INLA's stochastic optimizer can produce slight variations across different runs, the exact console output for the reported models is preserved in this repository.
