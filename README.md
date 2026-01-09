# AI-Based-Drone-Anomaly-Detection-Using-Machine-Learning-and-Deep-Learning-Models

Project Overview
This project presents an AI-driven framework for drone anomaly detection using telemetry and system-level data. The system is designed to identify abnormal drone behavior such as Denial-of-Service (DoS) attacks and hardware/software malfunctions, while distinguishing them from normal operational states.
Multiple machine learning and deep learning models are trained, optimized, evaluated, and explained using Explainable AI (XAI) techniques to ensure transparency and reliability.

Objectives
Detect anomalies in drone telemetry data
Classify drone behavior into Normal, DoS Attack, and Malfunction
Compare classical ML and deep learning approaches
Apply Explainable AI (SHAP, LIME, PDP) to interpret predictions
Provide reproducible and well-documented results

Dataset Description
The dataset consists of CSV files provided by the instructor:
Normal1.csv,Normal2.csv,Normal3.csv,Normal4.csv
Dos1.csv, Dos2.csv
Malfunction1.csv, Malfunction2.csv

Features
CPU usage and system performance metrics
Sensor readings (temperature, velocity, altitude)
Battery and RAM statistics
Time-based telemetry values
Target label (Label)


Project Structure
AI_Drone_Anomaly_Detection/
│
├── datasets/
│
│
├── notebooks/
│   ├── Part1_Data_Preprocessing.ipynb
│   ├── Part2_Model_Training.ipynb
│   ├── Part3_Model_Evaluation.ipynb
│   └── Part4_XAI_Analysis.ipynb
│
├── models/
│   ├── xgboost_model.pkl
│   ├── svm_model.pkl
│   ├── fnn_model.h5
│   ├── lstm_model.h5
│   └── cnn_model.h5
│
├── Visualization/
│
├── README/
│   ├── README.md
│
│
├── reports/
│   ├── Assignment Report
│   ├── XAI analysis 
│

Part 1: Data Preprocessing
Steps Performed
Merged multiple CSV files into a single dataset
Removed duplicates
Handled missing values using median imputation
Scaled numerical features using StandardScaler
Encoded labels

Generated extensive visualizations:
Distributions
Box plots
Correlation heatmaps
Time-series plots
Missing data heatmaps
Class distribution plots

Output

drone_data_cleaned.csv (ready for model training)

Part 2: Model Training
Models Implemented
LSTM – temporal dependency modeling
1D CNN – local temporal feature extraction
SVM – classical ML baseline
XGBoost – high-performance gradient boosting
Variational Autoencoder (VAE) – unsupervised anomaly detection
Feedforward Neural Network (FNN) – deep learning baseline

Hyperparameter Tuning
Grid Search CV (SVM)
Random Search CV (XGBoost, FNN, CNN, LSTM)
Early stopping where applicable

Saved Models

All trained models are saved for reproducibility.

Part 3: Model Evaluation
Evaluation Metrics
Accuracy
Precision, Recall, F1-score
Confusion Matrix
ROC-AUC

Learning Curves

Key Results
XGBoost achieved the highest overall performance
Deep learning models captured complex and temporal patterns
VAE useful for unsupervised anomaly detection

Part 4: Explainable AI (XAI)
Techniques Used
Feature Importance (XGBoost)
SHAP (global & local explanations)
LIME (instance-level explanations)
Partial Dependence Plots (PDP)

Key Insights

CPU usage, sensor temperature, velocity, and battery voltage are dominant predictors

Strong non-linear relationships exist

Feature interactions are critical in anomaly detection

Explanations align well with drone domain knowledge

Tools & Libraries
Python 3.10+
NumPy, Pandas
Matplotlib, Seaborn
Scikit-learn
TensorFlow / Keras
XGBoost
SHAP
LIME

How to Run
Upload all files to Google Colab
Run notebooks in order:
Part1 → Part2 → Part3 → Part4


Ensure required libraries are installed:

!pip install xgboost shap lime

Results & Deliverables

✔ Preprocessed dataset
✔ Trained models
✔ Evaluation metrics and plots
✔ XAI explanations
✔ Final written report (PDF)

Conclusion

This project demonstrates that AI-based anomaly detection can significantly enhance drone safety and reliability. Combining high-performing models with explainable AI techniques ensures both accuracy and trustworthiness, making the system suitable for real-world drone monitoring applications.
