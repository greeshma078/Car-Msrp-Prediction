import joblib
import pandas as pd
import logging
import os

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Load the trained pipeline
pipeline = joblib.load("models/final_pipeline.pkl")

def predict_price(input_dict):
    """
    Predict MSRP given a dictionary of car features.
    Logs input and prediction to logs/predictions.log
    """
    # Convert dict to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Normalize column names to match training
    input_df.columns = (
        input_df.columns.str.strip()
                        .str.lower()
                        .str.replace(" ", "_")
                        .str.replace("-", "_")
    )

    # Predict
    prediction = pipeline.predict(input_df)[0]

    # Log input and prediction
    logging.info(f"Input: {input_dict}, Prediction: {prediction}")

    return prediction

# Example usage (for testing only)
if __name__ == "__main__":
    sample = {
        "make": "BMW",
        "model": "1 Series",
        "year": 2011,
        "engine_hp": 300,
        "engine_cylinders": 6,
        "transmission_type": "AUTOMATIC",
        "driven_wheels": "rear wheel drive",
        "number_of_doors": 2,
        "market_category": "Luxury,Performance",
        "vehicle_size": "Compact",
        "vehicle_style": "Coupe",
        "highway_mpg": 26,
        "city_mpg": 19,
        "popularity": 3916
    }
    price = predict_price(sample)
    print(f"Predicted MSRP: ${price:,.2f}")
