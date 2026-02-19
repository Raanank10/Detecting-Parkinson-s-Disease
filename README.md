Detecting Parkinson's Disease with Machine Learning
🩺 Project Overview
This project leverages machine learning to detect Parkinson’s Disease (PD) using a dataset of biomedical voice measurements. As a Medical Engineer, I developed this project to demonstrate the intersection of clinical diagnostics and predictive analytics.

The goal is to differentiate healthy individuals from those with Parkinson's by analyzing vocal characteristics, providing a high-accuracy, non-invasive screening tool.

📊 Dataset Description
The dataset consists of 195 biomedical voice recordings from people with and without Parkinson's disease.

Target Feature (status): 1 for Parkinson's, 0 for Healthy.

Input Features: 22 vocal attributes, including:

Jitter & Shimmer: Measures of variation in fundamental frequency and amplitude.

HNR & NHR: Harmonics-to-Noise and Noise-to-Harmonics ratios.

DFE: Spread of fundamental frequency.

RPDE & D2: Nonlinear dynamical complexity measures.

🛠️ Technical Stack
Language: Python 3.x

Libraries:

XGBoost: Used for the core predictive modeling due to its performance on structured data.

Scikit-learn: For data scaling (MinMaxScaler), train-test splitting, and evaluation metrics.

Pandas & NumPy: For data manipulation and feature engineering.

🚀 Workflow
Data Preprocessing: Handled the imbalanced nature of the dataset and used MinMaxScaler to normalize features between -1 and 1 for better model convergence.

Feature Selection: Isolated the status column as the target variable and utilized all vocal measurements as predictors.

Model Training: Implemented an XGBClassifier (Extreme Gradient Boosting), an ensemble method known for high precision in classification tasks.

Evaluation: Measured performance using an accuracy score and a confusion matrix to minimize false negatives (critical in medical diagnostics).

📈 Results
Accuracy: Reached a high performance of ~94.8% (varies slightly by random seed).

Clinical Significance: The model effectively identifies the vocal "fingerprints" of PD, which are often subtle and difficult for the human ear to distinguish in early stages.

📂 Repository Structure
Detecting_Parkinsons_Disease.ipynb: The complete end-to-end Python code and analysis.

parkinsons.csv: The raw biomedical dataset.

👤 Author
Raanan Kelner

B.Sc. in Medical Engineering

Data & Business Analyst

LinkedIn | GitHub Portfolio
