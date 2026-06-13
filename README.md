# Car-Msrp-Prediction
Car MSRP Prediction is a machine learning project that estimates a car’s Manufacturer’s Suggested Retail Price using features like make, model, year, engine specs, fuel type, transmission, and popularity. It integrates preprocessing, RandomForest modeling, MLflow tracking, logging, and a Streamlit app for deployment.

##  Approach
- Data preprocessing with scikit-learn (handling missing values, encoding, scaling).
- Model training with RandomForestRegressor.
- Evaluation using R², MAE, RMSE.
- Experiment tracking with MLflow.
- Prediction logging for monitoring.
- Deployment via Streamlit app.

##  Results
- R²: 0.969  
- MAE: ~2974  
- RMSE: ~8567  

##  How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/greeshma078/Car-Msrp-Prediction.git
