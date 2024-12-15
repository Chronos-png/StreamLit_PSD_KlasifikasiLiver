# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Streamlit app
st.title("Liver Patient Dataset Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("indian_liver_patient.csv", type=["csv"])

if uploaded_file is not None:
    # Load the file
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Overview")
    st.dataframe(df.head())
    
    # Check for missing values
    st.write("### Missing Values Count")
    st.write(df.isnull().sum())
    
    # Fill missing values
    data_filled = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    st.write("### Missing Values After Filling")
    st.write(data_filled.isnull().sum())
    
    # Transform data
    data_transformed = data_filled.copy()
    data_transformed['Gender'] = data_transformed['Gender'].map({'Female': 0, 'Male': 1})
    data_transformed['Dataset'] = data_transformed['Dataset'].map({2: 0, 1: 1})
    data_transformed.drop(['Direct_Bilirubin', 'Albumin', 'Aspartate_Aminotransferase'], axis=1, inplace=True)
    st.write("### Transformed Dataset")
    st.dataframe(data_transformed.head())
    
    # Normalize the data
    X = data_transformed.drop('Dataset', axis=1)
    y = data_transformed['Dataset']
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Logistic Regression
    st.write("### Logistic Regression Model")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_test_pred = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Classification Report
    st.write("### Classification Report")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
