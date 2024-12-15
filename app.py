import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE

# Global Variables
scaler = StandardScaler()  # Define scaler globally
lr_model = None
rf_model = None
nn_model = None

# Streamlit UI
st.title("Liver Disease Prediction")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Demo", "Prediction"])

file_path = 'indian_liver_patient.csv'

def display_metrics(y_test, y_pred, title):
    # Function to display performance metrics (accuracy, precision, recall, etc.)
    ...

if menu == "Demo":
    try:
        # Load and process dataset
        df = pd.read_csv(file_path)
        # Data Preprocessing & Transformation
        ...

        # Transform Data
        X = data_transformasi.drop('Dataset', axis=1)
        y = data_transformasi['Dataset']

        # Fit the scaler on the training data (this part was missing before)
        x_normalized = scaler.fit_transform(X)  # Fit and transform on the training data
        x_normalized = pd.DataFrame(x_normalized, columns=X.columns)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

        # SMOTE
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train Models
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        nn_model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        nn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        nn_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    except FileNotFoundError:
        st.error(f"File {file_path} not found. Please make sure the file is in the same folder.")

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

        # Use the already fitted scaler to transform input data
        input_scaled = scaler.transform(input_data)  # This will work now since scaler is fitted

        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(input_scaled)[0]
        elif model_choice == "Random Forest":
            prediction = rf_model.predict(input_scaled)[0]
        else:
            prediction = (nn_model.predict(input_scaled) > 0.5).astype(int)[0][0]

        prediction_result = "Positive for Liver Disease" if prediction == 1 else "Negative for Liver Disease"

        st.write("### Prediction Result")
        st.write(prediction_result)
