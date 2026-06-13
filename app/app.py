import streamlit as st
import sys, os

# Fix import path first so Python can find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import training function
from src.train import train_model  

# Retrain if model file is missing
if not os.path.exists("models/final_pipeline.pkl"):
    st.warning("Model file not found. Training model... please wait.")
    train_model()
    st.success("Model trained and saved!")

# Import prediction function AFTER ensuring model exists
from src.predict import predict_price

# Streamlit UI
st.title("Car MSRP Prediction App")
st.write("Enter car details below to predict the Manufacturer's Suggested Retail Price (MSRP).")

# User inputs
make = st.text_input("Car Make", "BMW")
model_name = st.text_input("Car Model", "1 Series")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2011)
engine_hp = st.number_input("Engine HP", min_value=50, max_value=1000, value=300)
engine_cylinders = st.number_input("Engine Cylinders", min_value=2, max_value=16, value=6)
transmission_type = st.selectbox("Transmission Type", ["MANUAL", "AUTOMATIC"])
driven_wheels = st.selectbox("Driven Wheels", ["rear wheel drive", "front wheel drive", "all wheel drive"])
number_of_doors = st.number_input("Number of Doors", min_value=2, max_value=5, value=2)
market_category = st.text_input("Market Category", "Luxury,Performance")
vehicle_size = st.selectbox("Vehicle Size", ["Compact", "Midsize", "Large"])
vehicle_style = st.selectbox("Vehicle Style", ["Coupe", "Sedan", "SUV", "Convertible"])
highway_mpg = st.number_input("Highway MPG", min_value=5, max_value=150, value=26)
city_mpg = st.number_input("City MPG", min_value=5, max_value=100, value=19)
popularity = st.number_input("Popularity", min_value=1, max_value=6000, value=3916)

# Collect inputs into dictionary
input_dict = {
    "make": make,
    "model": model_name,
    "year": year,
    "engine_hp": engine_hp,
    "engine_cylinders": engine_cylinders,
    "transmission_type": transmission_type,
    "driven_wheels": driven_wheels,
    "number_of_doors": number_of_doors,
    "market_category": market_category,
    "vehicle_size": vehicle_size,
    "vehicle_style": vehicle_style,
    "highway_mpg": highway_mpg,
    "city_mpg": city_mpg,
    "popularity": popularity
}

# Prediction button
if st.button("Predict MSRP"):
    price = predict_price(input_dict)
    st.success(f"Predicted MSRP: ${price:,.2f}")
