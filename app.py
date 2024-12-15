import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Streamlit UI
st.title("Liver Disease Prediction")

# Load File from Local Folder
file_path = 'indian_liver_patient.csv'
try:
    df = pd.read_csv(file_path)
    st.write("## Dataset Preview")
    st.dataframe(df.head())

    # Missing Value Check
    st.write("## Missing Value Check")
    st.write(df.isnull().sum())

    # Handle Missing Values
    data_terisi = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    st.write("## Data After Handling Missing Values")
    st.write(data_terisi.isnull().sum())

    # Transform Data
    data_transformasi = data_terisi.copy()
    data_transformasi['Gender'] = data_transformasi['Gender'].map({'Female': 0, 'Male': 1})
    data_transformasi['Dataset'] = data_transformasi['Dataset'].map({2: 0, 1: 1})
    # data_transformasi.drop(['Direct_Bilirubin', 'Albumin', 'Aspartate_Aminotransferase'], axis=1, inplace=True)

    st.write("## Data After Transformation")
    st.dataframe(data_transformasi.head())

    # Normalization
    X = data_transformasi.drop('Dataset', axis=1)
    y = data_transformasi['Dataset']
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(X)
    x_normalized = pd.DataFrame(x_normalized, columns=X.columns)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

    # SMOTE
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Logistic Regression
    st.write("## Logistic Regression")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_y_test_hat = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_y_test_hat)
    st.write(f"Test Accuracy: {lr_accuracy * 100:.2f}%")
    st.text(classification_report(y_test, lr_y_test_hat))

    # Random Forest
    st.write("## Random Forest")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_y_test_hat = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_test_hat)
    st.write(f"Test Accuracy: {rf_accuracy * 100:.2f}%")
    st.text(classification_report(y_test, rf_y_test_hat))

    # Neural Network
    st.write("## Neural Network")
    nn_model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer=Adam(),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    nn_y_pred = (nn_model.predict(X_test) > 0.5).astype(int)
    nn_accuracy = accuracy_score(y_test, nn_y_pred)
    st.write(f"Test Accuracy: {nn_accuracy * 100:.2f}%")
    st.text(classification_report(y_test, nn_y_pred))

    # Visualization
    st.write("## Confusion Matrix")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(confusion_matrix(y_test, lr_y_test_hat), annot=True, fmt='d', ax=ax[0], cmap='Blues')
    ax[0].set_title('Logistic Regression')

    sns.heatmap(confusion_matrix(y_test, rf_y_test_hat), annot=True, fmt='d', ax=ax[1], cmap='Greens')
    ax[1].set_title('Random Forest')

    sns.heatmap(confusion_matrix(y_test, nn_y_pred), annot=True, fmt='d', ax=ax[2], cmap='Reds')
    ax[2].set_title('Neural Network')

    st.pyplot(fig)
except FileNotFoundError:
    st.error(f"File {file_path} not found. Please make sure the file is in the same folder.")
