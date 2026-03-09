import joblib 
import numpy as np
import pandas as pd

model = joblib.load("../models/traffic_model.pkl")

print("Model loaded successfully");



def predict_traffic(hour, day, temperature, precipitation, cloudcover, windspeed, is_holiday):

    features = pd.DataFrame([{
        "hour": hour,
        "day": day,
        "temperature": temperature,
        "precipitation": precipitation,
        "cloudcover": cloudcover,
        "windspeed": windspeed,
        "is_holiday": is_holiday
    }])

    prediction = model.predict(features)[0]
    
    traffic_map = {
        0: "Low Traffic",
        1: "Medium Traffic",
        2: "High Traffic"
    }

    return traffic_map[prediction]


if __name__ == "__main__":
    result = predict_traffic(
        hour=9,
        day=1,
        temperature=28,
        precipitation=0,
        cloudcover=60,
        windspeed=15,
        is_holiday=0
    )

    print("Predicted Traffic Level:", result)