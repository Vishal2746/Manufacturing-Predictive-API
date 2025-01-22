# Manufacturing Predictive API

This API predicts machine downtime or production defects using a machine learning model trained on manufacturing data. It allows users to upload data, train the model, and make predictions using simple API endpoints.

### Features:
- **/upload**: Upload a CSV file containing manufacturing data.
- **/train**: Train a machine learning model on the uploaded data.
- **/predict**: Make predictions on new data to predict machine downtime.

## Setup Instructions

### Prerequisites:
- Python 3.x
- Libraries: Flask, scikit-learn, pandas, imbalanced-learn, XGBoost

Start the Flask application:
python app.py

The API will run locally at http://127.0.0.1:5000


### **Example Requests**

```markdown
## Example Requests

### 1. **Upload Data**:
   - Method: **POST**
   - URL: `http://127.0.0.1:5000/upload`
   - Body: **form-data** (Upload `sample_data.csv` file)

### 2. **Train Model**:
   - Method: **POST**
   - URL: `http://127.0.0.1:5000/train`

### 3. **Make Prediction**:
   - Method: **POST**
   - URL: `http://127.0.0.1:5000/predict`
   - Body: **raw** (JSON):
     ```json
     {
         "Temperature": 80,
         "Run_Time": 120
     }
     ```
   - Example Response:
     ```json
     {
         "Downtime": "Yes",
         "Confidence": 0.85
     }
     ```


**Once the server is running, you can test all routes (`/upload`, `/train`, `/predict`) using tools like **Postman** or **cURL**.



