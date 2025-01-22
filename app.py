from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the saved model
with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Welcome to the Manufacturing Predictive API!"

@app.route('/upload', methods=['POST'])
def upload_data():
    os.makedirs('data', exist_ok=True)
    file = request.files['file']
    file_path = os.path.join('data', file.filename)
    file.save(file_path)
    return jsonify({"message": f"Data uploaded successfully as {file.filename}!"})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

@app.route('/train', methods=['POST'])
def train_model():
    file_path = os.path.join('data', os.listdir('data')[0])
    data = pd.read_csv(file_path)
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']
    model.fit(X, y)
    with open('model/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    return jsonify({"message": "Model trained successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    X_new = [[input_data['Temperature'], input_data['Run_Time']]]
    prediction = model.predict(X_new)
    confidence = model.predict_proba(X_new).max()
    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": confidence
    })

# Print all available routes for debugging
print("Registered Routes:")
for rule in app.url_map.iter_rules():
    print(rule)

if __name__ == '__main__':
    app.run(debug=True)
