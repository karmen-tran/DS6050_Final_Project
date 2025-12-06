# DS6050_Final_Project

# Interpretable AI for Clinical Risk Prediction

This project provides a structured pipeline for analyzing health-related datasets using classical and neural network models. It focuses on logistic regression, MLP (neural networks), and decision tree baselines, with modules for metrics, interpretability, and data preprocessing.

## Datasets

The project uses the following publicly available datasets from the UCI Machine Learning Repository:

- **Heart Disease** – Predicting the presence of heart disease. [UCI Link](https://archive.ics.uci.edu/ml/datasets/heart+Disease)  
- **Chronic Kidney Disease** – Clinical data for predicting chronic kidney disease. [UCI Link](https://archive.ics.uci.edu/ml/datasets/chronic_kidney_disease)  
- **CDC Diabetes** – Diabetes risk and clinical indicators. [UCI Link](https://archive.ics.uci.edu/ml/datasets/diabetes)  

Each dataset is preprocessed to handle missing values, encode categorical features, and manage class imbalance. Both binary and multiclass classification tasks are supported.

## Project Structure

- **notebooks/**: Jupyter notebooks for experiments and visualization.  
- **src/**: Core modules including data loading, model training, metrics, and interpretability.  
- **requirements.txt**: Python dependencies.  

## Features

- Data preprocessing: handle missing values, encode categorical features, stratified splits.  
- Models: logistic regression, MLP (NumPy), decision tree baseline.  
- Metrics: ROC-AUC, F1 scores, balanced accuracy, recall.  
- Interpretability: feature importance from logistic regression, input saliency for MLP.  
- Visualization: class imbalance, performance comparison, and feature analysis plots.  

## Installation

Create a Python environment and install dependencies from `requirements.txt`.  

## Usage

Use the notebooks to load datasets, train models, evaluate metrics, and generate plots. Core logic is in `src/`, while notebooks are mainly for experimentation and visualization.

