# Crime Data Classification Hackathon Project

This repository contains the code and documentation for a text classification project aimed at categorizing crime-related descriptions into predefined categories, subcategories, and victim types. The project was developed as part of a hackathon, with the objective of creating a robust and scalable NLP model that aligns with the Government of India's guidelines for data handling and categorization.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Selection](#model-selection)
  - [Training and Tuning](#training-and-tuning)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Deployment](#deployment)
- [Future Work](#future-work)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Project Overview
The purpose of this project is to automate the classification of crime-related text descriptions using machine learning and NLP techniques. The classification process is divided into three levels: category, subcategory, and victim type, which are defined according to government regulations. The model has been trained to achieve high accuracy and reliability, making it suitable for real-world applications in government systems to enhance data organization and decision-making.

## Dataset
The dataset contains anonymized crime reports with textual descriptions and associated labels for categories, subcategories, and victim types. Some preprocessing steps were applied to handle imbalanced classes and missing data, ensuring a clean and balanced dataset for training.

## Methodology

### Data Preprocessing
- Missing subcategory labels were filled based on corresponding categories.
- Data cleaning involved tokenization, stopword removal, and lemmatization.
- Addressed class imbalance using SMOTE and other techniques to ensure balanced representation across labels.

### Feature Engineering
- Applied TF-IDF (Term Frequency-Inverse Document Frequency) to transform text data into numerical features.
- Considered alternative techniques such as Word2Vec and BERT, but TF-IDF proved effective in balancing interpretability and performance.

### Model Selection
- We selected a multi-output Random Forest classifier for its ensemble learning capabilities and robust performance across multi-label tasks.
- Other models like Support Vector Machines (SVM) and Logistic Regression were considered but did not perform as well as Random Forest in terms of accuracy and interpretability.

### Training and Tuning
- Hyperparameter tuning was performed using GridSearchCV to optimize model parameters.
- Multi-output functionality was incorporated to handle the hierarchical classification structure.

### Evaluation
- Model evaluation was based on multiple metrics, including accuracy, precision, recall, and F1-score, for each classification level.
- Achieved 98.24% accuracy for category classification, 96.93% for subcategory classification, and 90.34% for victim type classification.

## Results
The model demonstrates strong performance across all classification levels:
- **Category**: Accuracy - 98.24%, Precision - 99.49%, Recall - 98.24%, F1-Score - 98.77%
- **Subcategory**: Accuracy - 96.93%, Precision - 97.92%, Recall - 96.93%, F1-Score - 97.29%
- **Victim Type**: Accuracy - 90.34%, Precision - 96.05%, Recall - 90.34%, F1-Score - 92.83%




## Future Work
Future improvements include:
- Implementing transformer-based models (e.g., BERT) for deeper contextual understanding.
- Expanding the dataset with additional data sources.
- Automating model retraining to handle evolving data patterns and ensure long-term accuracy.

## Requirements
To run this project, you'll need the following libraries:
- `scikit-learn`
- `pandas`
- `numpy`
- `nltk`
- `wordcloud`
- `mlflow`
## Model and Artifacts

All the models and artifacts related to this project are stored in the following Google Drive folder:

[**Google Drive Models and Artifacts**](https://drive.google.com/drive/folders/10NBX9fFJINhzne_bCQRPWSz6cLEUhOfh?usp=drive_link)

You can access and download the necessary files from the link above.
A
