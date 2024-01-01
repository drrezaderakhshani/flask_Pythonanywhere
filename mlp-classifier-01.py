
# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
file_path = r'C:\Users\r-derakhshani\Desktop\flask_Pythonanywhere\data\ANN-Input.xlsx'
df = pd.read_excel(file_path)

# Step 3: Data Preprocessing
# 3.1 Drop the identifier column
df.drop('identifier', axis=1, inplace=True)


# 3.2 Encode categorical variables
label_encoder = LabelEncoder()
categorical_columns = ['province', 'vehicle_type', 'customer_type', 'ly_insurer',
                        'pelak_type', 'life_history', 'passenger_history', 'financial_history']

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# 3.3 One-Hot Encoding for variables with more than two categories
df = pd.get_dummies(df, columns=['province', 'ly_insurer', 'pelak_type', 'life_history', 
                                  'passenger_history', 'financial_history'], drop_first=True)

# 3.4 Split the dataset into features (X) and target (y)
X = df.drop('cluster_name', axis=1)  # 'cluster_name' is the target variable
y = df['cluster_name']


# 3.5 Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3.6 Standardize the features (optional but recommended for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train the MLP classifier
model = MLPClassifier(hidden_layer_sizes=(11,), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predictions and Evaluation
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Additional: You can print a classification report for more detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import pickle
pickle.dump(model,open('model.pkl','wb'))




 



