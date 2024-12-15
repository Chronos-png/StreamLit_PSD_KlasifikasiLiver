import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Streamlit UI
st.title("Liver Disease Prediction")

# Sidebar menu
menu = st.sidebar.selectbox("Menu", ["Demo", "Prediction"])

file_path = 'indian_liver_patient.csv'

def display_metrics(y_test, y_pred, title):
    fig, ax = plt.subplots(1, figsize=(15, 5))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_title(title)

    st.pyplot(fig)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"### Performance Metrics")
    print(f"- **Accuracy**: {accuracy * 100:.2f}%")
    print(f"- **Precision**: {precision:.2f}")
    print(f"- **Recall**: {recall:.2f}")
    print(f"- **F1 Score**: {f1:.2f}")

if menu == "Demo":
    try:
        # Load File from Local Folder
        df = pd.read_csv(file_path)
        st.write("## Dataset Mentahan")
        st.dataframe(df)

        # Missing Value Check
        st.write("## Missing Value Check")
        st.write(df.isnull().sum())

        # Handle Missing Values
        col1,col2 = st.columns(2)
        with col1:
            data_terisi = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
            st.subheader("Data Setelah Missing Value Diisi ( Mean )")
            st.write(data_terisi.isnull().sum())
        with col2:
            st.subheader("Data Sekarang")
            st.dataframe(df)

        # Transform Data
        data_transformasi = data_terisi.copy()
        data_transformasi['Gender'] = data_transformasi['Gender'].map({'Female': 0, 'Male': 1})
        data_transformasi['Dataset'] = data_transformasi['Dataset'].map({2: 0, 1: 1})

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

        st.write("### Model Architecture")
        st.write(lr_model)
        display_metrics(y_test, lr_y_test_hat, 'Logistic Regression')

        # Random Forest
        st.write("## Random Forest")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_y_test_hat = rf_model.predict(X_test)

        st.write("### Model Architecture")
        st.write(rf_model)
        display_metrics(y_test, rf_y_test_hat, 'Random Forest')

        # Neural Network
        st.write("## JST Backpropagation")
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

        st.write("### Model Architecture")
        for i, layer in enumerate(nn_model.layers):
            layer_config = layer.get_config()
            st.write(f"Layer {i+1}:")
            st.write(f"- Name: {layer_config['name']}")
            st.write(f"- Units: {layer_config.get('units', 'N/A')}")
            st.write(f"- Activation: {layer_config.get('activation', 'N/A')}")
        display_metrics(y_test, nn_y_pred, 'JST Backpropagation')

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

        input_scaled = scaler.transform(input_data)

        nn_prediction = (nn_model.predict(input_scaled) > 0.5).astype(int)[0][0]
        prediction_result = "Positive for Liver Disease" if nn_prediction == 1 else "Negative for Liver Disease"

        st.write("### Prediction Result")
        st.write(prediction_result)
