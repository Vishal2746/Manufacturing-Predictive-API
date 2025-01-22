import pandas as pd

# Load the dataset
data = pd.read_csv('data/sample_data.csv')
print(data.head())  # Display the first 5 rows of the dataset

from sklearn.model_selection import train_test_split

# Features (input variables)
X = data[['Temperature', 'Run_Time']] 

# Target (output variable)
y = data['Downtime_Flag']

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Test the model on the testing set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

import os
import pickle

os.makedirs('model', exist_ok=True)

# Save the trained model
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved at: model/model.pkl")
