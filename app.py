import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import pickle

# Streamlit UI
st.title("Liver Disease Prediction")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Demo", "Prediction"])


def display_metrics(y_test, y_pred, title):
    fig, ax = plt.subplots(1, figsize=(15, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)
    st.pyplot(fig)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"### Performance Metrics")
    st.write(f"- **Accuracy**: {accuracy * 100:.2f}%")
    st.write(f"- **Precision**: {precision:.2f}")
    st.write(f"- **Recall**: {recall:.2f}")
    st.write(f"- **F1 Score**: {f1:.2f}")


if menu == "Demo":
    try:
        # Load File from Local Folder
        df = pd.read_csv('indian_liver_patient.csv')
        st.write("## Dataset Mentahan")
        st.dataframe(df)

        # Missing Value Check
        data_terisi = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
        data_transformasi = data_terisi.copy()
        data_transformasi['Gender'] = data_transformasi['Gender'].map({'Female': 0, 'Male': 1})
        data_transformasi['Dataset'] = data_transformasi['Dataset'].map({2: 0, 1: 1})

        # Normalization
        X = data_transformasi.drop('Dataset', axis=1)
        y = data_transformasi['Dataset']
        scaler = StandardScaler()
        x_normalized = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

        # SMOTE
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Logistic Regression
        st.subheader("Logistic Regression")
        model_path = 'lr_model65.pkl'
        with open(model_path, 'rb') as file:
            lr_model = pickle.load(file)
        lr_y_pred = lr_model.predict(X_test)
        display_metrics(y_test, lr_y_pred, 'Logistic Regression')

        # Random Forest
        st.subheader("Random Forest")
        model_path = 'rf_model73.pkl'
        with open(model_path, 'rb') as file:
            rf_model = pickle.load(file)
        rf_y_pred = rf_model.predict(X_test)
        display_metrics(y_test, rf_y_pred, 'Random Forest')

        # Neural Network
        st.subheader("Neural Network (Backpropagation)")
        model_path = 'model_keras.h5'
        nn_model = load_model(model_path)
        nn_y_pred = (nn_model.predict(X_test) > 0.5).astype(int).flatten()
        display_metrics(y_test, nn_y_pred, 'Neural Network (Backpropagation)')

    except FileNotFoundError as e:
        st.error(f"File not found. Make sure all required files are available: {e}")

elif menu == "Prediction":
    st.write("## Prediction Form")

    # Form Input
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, value=1.0)
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, value=200)
        alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=0, value=20)
        total_proteins = st.number_input("Total Proteins", min_value=0.0, value=6.5)
        albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, value=1.0)
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "JST Backpropagation"])
        submit_button = st.form_submit_button("Predict")

    # Prediction Process
    if submit_button:
        input_data = pd.DataFrame({
            "Gender": [1 if gender == "Male" else 0],
            "Age": [age],
            "Total_Bilirubin": [total_bilirubin],
            "Alkaline_Phosphotase": [alkaline_phosphotase],
            "Alanine_Aminotransferase": [alanine_aminotransferase],
            "Total_Proteins": [total_proteins],
            "Albumin_and_Globulin_Ratio": [albumin_and_globulin_ratio]
        })

        scaler = StandardScaler()
        # Logistic Regression
        model_path = 'lr_model65.pkl'
        with open(model_path, 'rb') as file:
            lr_model = pickle.load(file)

        # Random Forest
        model_path = 'rf_model73.pkl'
        with open(model_path, 'rb') as file:
            rf_model = pickle.load(file)

        # Neural Network
        model_path = 'model_keras.h5'
        nn_model = load_model(model_path)

        input_scaled = scaler.transform(input_data)

        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(input_scaled)[0]
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(input_scaled)[0]
        else:
            prediction = (nn_model.predict(input_scaled) > 0.5).astype(int)[0][0]

        prediction_result = "Positive for Liver Disease" if prediction == 1 else "Negative for Liver Disease"
        st.write("### Prediction Result")
        st.write(prediction_result)
