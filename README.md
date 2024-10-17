# Machine Learning Assessments

This repository contains materials for assessing beginner-level machine learning skills, including three different types of assessments and corresponding datasets.

## Overview

After completing the Comprehensive Machine Learning Course, participants are given the opportunity to choose one of three assessments to demonstrate their skills. Each assessment focuses on a different area of machine learning:

1. Text Classification
2. Regression Analysis
3. Clustering

## Assessments

### 1. Text Classification
- **Objective**: Classify customer reviews into positive or negative sentiment.
- **Skills Tested**: Natural Language Processing, binary classification, feature engineering with text data.
- **Dataset**: [Customer Reviews](datasets/customer_reviews.csv)
- **Notebook**: [Text Classification Assessment](assessments/text_classification.ipynb)
- **Key Notebook**: [Text Classification Assessment Key](assessments/text_classification_key.ipynb)

### 2. Regression Analysis
- **Objective**: Predict house prices based on various features.
- **Skills Tested**: Data preprocessing, feature selection, regression modeling, model evaluation.
- **Dataset**: [House Prices](datasets/house_prices.csv)
- **Notebook**: [Regression Analysis Assessment](assessments/regression_analysis.ipynb)
- **Key Notebook**: [Regression Analysis Assessment Key](assessments/regression_analysis_key.ipynb)

### 3. Clustering
- **Objective**: Identify customer segments based on their purchasing behavior.
- **Skills Tested**: Unsupervised learning, data visualization, feature scaling, cluster analysis.
- **Dataset**: [Customer Data](datasets/customer_data.csv)
- **Notebook**: [Clustering Assessment](assessments/clustering.ipynb)
- **Key Notebook**: [Clustering Assessment Key](assessments/clustering_key.ipynb)

## Repository Structure

```
ml-assessments/
│
├── README.md
├── assessments/
│   ├── text_classification.ipynb
│   ├── regression_analysis.ipynb
│   ├── clustering.ipynb
│   ├── text_classification_key.ipynb
│   ├── regression_analysis_key.ipynb
│   ├── clustering_key.ipynb
│
├── datasets/
│   ├── customer_reviews.csv
│   ├── house_prices.csv
│   ├── customer_data.csv
│
└── email_templates/
    └── assessment_choice_email.md
```

## Usage

1. Send the assessment choice email to participants (found in `email_templates/assessment_choice_email.md`).
2. Based on their choice, provide the participant with the appropriate Jupyter notebook and dataset.
3. Participants should complete the assessment within the given timeframe.
4. Use the `*_key.ipynb` notebooks for grading and providing feedback.

## Guidelines for Participants

- Choose one assessment that best aligns with your interests or strengths.
- Complete the assessment within one week of receiving the materials.
- Ensure all code cells are executed and outputs are visible in your submitted notebook.
- Provide explanations and interpretations where requested in the notebook.

## Feedback and Support

For any questions or issues, please open an issue in this repository or contact the course mentor directly.

Good luck to all participants!
