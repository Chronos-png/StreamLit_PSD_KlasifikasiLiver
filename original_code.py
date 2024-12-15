#Library
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

#Load File
lokasi_file = 'indian_liver_patient.csv'
df = pd.read_csv(lokasi_file)
df.head()

#MISSING VALUE CHECK
df.isnull().sum()
data_terisi = df.apply(lambda x: x.fillna(x.mode()[0]), axis=0)
data_terisi.isnull().sum()

#TRANSFORM
data_transformasi = data_terisi.copy()
data_transformasi['Gender'] = data_transformasi['Gender'].map({'Female': 0, 'Male': 1})
data_transformasi['Dataset'] = data_transformasi['Dataset'].map({2: 0, 1 : 1})
data_transformasi.drop(['Direct_Bilirubin', 'Albumin', 'Aspartate_Aminotransferase'], axis=1, inplace=True)
data_transformasi.head()

#NORMALIZATION
data_resize = data_transformasi.copy()
X = data_resize.drop('Dataset', axis=1)
y = data_resize['Dataset']
scaler = StandardScaler()
x_normalized = scaler.fit_transform(X)
x_normalized = pd.DataFrame(x_normalized)
x_normalized.columns = X.columns

#SET
X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=42)

#SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

#LR
lr_model = LogisticRegression()
lr_XTest = X_train.copy()
lr_YTest = y_train.copy()
lr_model.fit(lr_XTest, lr_YTest)
lr_y_train_hat = lr_model.predict(X_train)
lr_y_test_hat = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_y_test_hat)
print(f"Test Accuracy: {lr_accuracy * 100:.2f}%")

# Menampilkan model Logistic Regression
print(lr_model)

# Kinerja pada data pelatihan
print('Train performance')
print('-------------------------------------------------------')
print(classification_report(y_train, lr_y_train_hat))

# Kinerja pada data pengujian
print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, lr_y_test_hat))

#RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_XTest = X_train.copy()
rf_YTest = y_train.copy()
rf_model.fit(rf_XTest, rf_YTest)
rf_y_train_hat = rf_model.predict(X_train)
rf_y_test_hat = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_y_test_hat)
print(f"Test Accuracy: {rf_accuracy * 100:.2f}%")

# Menampilkan model Random Forest
print(rf_model)

# Kinerja pada data pelatihan
print('Train performance')
print('-------------------------------------------------------')
print(classification_report(y_train, rf_y_train_hat))

# Kinerja pada data pengujian
print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, rf_y_test_hat))

#BACKPROP...
nn_model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')])
nn_model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=["accuracy"])
nn_XTest = X_train.copy()
nn_YTest = y_train.copy()
nn_model.fit(nn_XTest, nn_YTest,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, y_test))

nn_y_pred = (nn_model.predict(X_test) > 0.5).astype(int)

# Menghitung akurasi untuk data uji
nn_accuracy = accuracy_score(y_test, nn_y_pred)
print(f"Test Accuracy: {nn_accuracy * 100:.2f}%")

# Menampilkan model Neural Network
print(nn_model)

# Kinerja pada data pelatihan (optional, jika Anda ingin menilai data pelatihan)
nn_y_train_hat = (nn_model.predict(X_train) > 0.5).astype(int)
print('Train performance')
print('-------------------------------------------------------')
print(classification_report(y_train, nn_y_train_hat))

# Kinerja pada data pengujian
print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, nn_y_pred))