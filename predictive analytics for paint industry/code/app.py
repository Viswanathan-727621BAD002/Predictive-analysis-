from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np  # Add numpy for array conversion

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("C:\Users\Viswa\OneDrive - Dr.MCET\Desktop\predictive analytics for paint industry\UpdatedData.xlsx")

# Assuming 'Total Volume' is your target variable column
# Select the feature columns
X = df.loc[:, 'XTVmbianceFinishes_vol':'JUHUXGuardian_pts']
# Create the target variable
y = df['Total Volume']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data:
        features = data['features']  # Assuming features are passed as a list in the JSON request
        # Convert features to DataFrame
        features_df = pd.DataFrame([features], columns=X.columns)  # Corrected DataFrame creation
        # Predict on the provided features
        predicted_volume = model.predict(features_df)
        return jsonify({'predicted_volume': predicted_volume.tolist()})
    else:
        return jsonify({'error': 'No data provided'})

if __name__ == '__main__':
    app.run(debug=True)
