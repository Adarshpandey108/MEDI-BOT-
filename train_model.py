import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
dataset_folder = 'datasets'
model_folder = 'model'
os.makedirs(model_folder, exist_ok=True)

# Load datasets
symptom_df = pd.read_csv(os.path.join(dataset_folder, 'diseases_and_symptoms.csv'))
precaution_df = pd.read_csv(os.path.join(dataset_folder, 'disease_precaution.csv'))

# Merge symptoms into a list per row
symptom_cols = [col for col in symptom_df.columns if col.startswith('Symptom_')]
symptom_df['All_Symptoms'] = symptom_df[symptom_cols].values.tolist()
symptom_df['All_Symptoms'] = symptom_df['All_Symptoms'].apply(lambda x: [i for i in x if pd.notna(i)])

# Convert symptoms into binary feature vectors
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptom_df['All_Symptoms'])
y = symptom_df['Disease']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and the symptom binarizer
joblib.dump(model, os.path.join(model_folder, 'disease_predictor_model.pkl'))
joblib.dump(mlb, os.path.join(model_folder, 'symptom_binarizer.pkl'))

print("✅ Model and binarizer saved successfully!")
