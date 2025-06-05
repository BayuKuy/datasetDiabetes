import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load data
url = 'https://raw.githubusercontent.com/BayuKuy/datasetDiabetes/main/diabetes.csv'
df = pd.read_csv(url)

# Heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi Fitur")
plt.show()

# Histogram distribusi fitur numerik
df.hist(bins=20, figsize=(14, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribusi Fitur Numerik", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

# Distribusi target
sns.countplot(x='Outcome', data=df)
plt.title("Distribusi Kelas Target (Diabetes)")
plt.xlabel("Outcome (0: Tidak Diabetes, 1: Diabetes)")
plt.ylabel("Jumlah")
plt.show()

# Pisahkan fitur dan target
X = df.drop('Outcome', axis=1)
y = df['Outcome']



# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standarisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversampling SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Naive Bayes
model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)

# Prediksi
y_pred = model.predict(X_test_scaled)

# Evaluasi
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
