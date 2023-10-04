from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from WindTurbinePipeline import WindTurbinePipeline 
from psycopg2 import connect, sql
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

load_dotenv()

print("AIIIIICI" + os.getcwd())

# Initialize your model
pipeline = WindTurbinePipeline(
    model_weights_path='/app/model/model.keras',
    scaler='/app/scaler/scaler_filename.pkl'
)

# Initialize DB connection
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


feature_name_mapping = {
    'WindSpeedAvg': 'Wind Speed avg',
    'RotorSpeedRpmAvg': 'Rotor Speed rpm avg',
    'ActivePowerAvg': 'Active Power avg',
    'NacellePositionAvg': 'Nacelle Position avg',
    'Feature1': 'Feature 1',
    'Feature3': 'Feature 3',
    'Feature7': 'Feature 7',
    'Feature28': 'Feature 28',
    'DaySin': 'Day sin',
    'DayCos': 'Day cos',
    'YearSin': 'Year sin',
    'YearCos': 'Year cos',
    'HourSin': 'hour sin',
    'HourCos': 'hour cos',
    'MinuteSin': 'minute sin',
    'MinuteCos': 'minute cos'
}



@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        df = pd.DataFrame(data)
        pipeline.train(df, epoch=50)
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_values():
    try:
        data = request.json
        df = pd.DataFrame(data)
        predictions = pipeline.predict(df)
        return jsonify({'status': 'success', 'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    


@app.route('/predict_future', methods=['GET'])
def predict_future():
    try:
        # SQL to fetch the last 48 records
        fetch_sql = "SELECT * FROM \"TurbineData\" ORDER BY id DESC LIMIT 64;"
        
        # Execute SQL and fetch data into Pandas DataFrame
        df = pd.read_sql(fetch_sql, engine)
        df.drop('id', axis=1, inplace=True)
        df.rename(columns=feature_name_mapping, inplace=True)

        # Ensure we have at least 48 rows before proceeding to prediction
        if len(df) < 64:
            return jsonify({'status': 'error', 'message': 'Not enough data for prediction'}), 400

        # Perform prediction
        predictions = pipeline.predict(df, is_preprocessed=True)

        return jsonify({'status': 'success', 'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
