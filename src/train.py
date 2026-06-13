import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.preprocessing import build_preprocessor

# Step 1: Load dataset
df = pd.read_csv("data/car_MSRP.csv")

# Step 2: Normalize column names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
              .str.replace("-", "_")
)

print("Normalized columns:", df.columns.tolist())

# Step 3: Define features and target
X = df[[
    "make","model","year","engine_hp","engine_cylinders",
    "transmission_type","driven_wheels","number_of_doors",
    "market_category","vehicle_size","vehicle_style",
    "highway_mpg","city_mpg","popularity"
]]
y = df["msrp_(manufacturer's_suggested_retail_price)"]

# Step 4: Build preprocessor
preprocessor = build_preprocessor(X)

# Step 5: Build pipeline
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Step 6: Train pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)

# Step 7: Evaluate model
y_pred = pipeline.predict(X_test)
print(" Model Evaluation:")
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Step 8: Log with MLflow
import numpy as np
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("rmse", np.sqrt(mean_squared_error(y_test, y_pred)))
    mlflow.sklearn.log_model(pipeline, name="model")

# Step 9: Save pipeline
joblib.dump(pipeline, "models/final_pipeline.pkl")
print(" Pipeline trained, evaluated, logged, and saved as models/final_pipeline.pkl")
