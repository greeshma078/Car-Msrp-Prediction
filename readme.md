# Car MSRP Prediction

## Problem Statement
Predict the Manufacturer's Suggested Retail Price (MSRP) of cars based on features like make, model, year, engine specs, MPG, and popularity.

## Approach
- Data preprocessing with scikit-learn (handling missing values, scaling, one-hot encoding).
- Model training using RandomForestRegressor.
- Experiment tracking with MLflow.
- Logging predictions for monitoring.
- Deployment via Streamlit app.

## Results
- R²: 0.969
- MAE: ~2974
- RMSE: ~8567

##  How to Run
1. Train the model:
   ```bash
   python -m src.train
