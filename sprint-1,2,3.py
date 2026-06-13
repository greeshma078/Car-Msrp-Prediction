# sprint-1.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 1: Load dataset
df = pd.read_csv("car_MSRP.csv")

# Step 2: Standardize column names
df.rename(columns={"MSRP (Manufacturer's suggested retail Price)": "msrp"}, inplace=True)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(" Dataset loaded")

# Step 3: Initial inspection
print("\n--- Head ---"); print(df.head())
print("\n--- Tail ---"); print(df.tail())
print("\n--- Info ---"); print(df.info())
print("\n--- Describe ---"); print(df.describe())
print("\n--- Shape ---"); print(df.shape)
print("\n--- Columns ---"); print(df.columns)

# Extra checks
print("\n--- Unique values per column ---")
for col in df.columns: print(col, df[col].nunique())
print("\n--- Missing values per column ---"); print(df.isnull().sum())

# Step 4: Handle duplicates
df = df.drop_duplicates()
print(" Duplicates dropped. New shape:", df.shape)

# Step 5: Handle missing values
for col in df.select_dtypes(include=['float64','int64']).columns:
    df.loc[:, col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=['object']).columns:
    df.loc[:, col] = df[col].fillna(df[col].mode()[0])
print(" Missing values handled")

# Step 6: EDA (plots)
plt.figure(figsize=(8,6))
sns.histplot(df['msrp'], bins=50, kde=True)
plt.title("Distribution of MSRP"); plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='engine_hp', y='msrp', data=df)
plt.title("Engine HP vs MSRP"); plt.show()

corr = df.corr(numeric_only=True)
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.show()

# Step 7: Outlier removal
Q1, Q3 = df['msrp'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df = df[(df['msrp'] >= Q1 - 1.5*IQR) & (df['msrp'] <= Q3 + 1.5*IQR)]
print(" Outliers removed. New shape:", df.shape)

# Step 8: Encoding + Scaling
X = df.drop('msrp', axis=1)
y = df['msrp']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64','float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

print("\n Sprint-1 complete")
print("Training set:", X_train.shape, "Testing set:", X_test.shape)


# sprint-2.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Baseline Model
print("\n--- Baseline Model: Linear Regression ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
print(f"MAE={mean_absolute_error(y_test, y_pred_lin):.2f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred_lin)):.2f}, R²={r2_score(y_test, y_pred_lin):.4f}")

# Step 2: Train Multiple Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []
print("\n--- Regression Models Evaluation ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"\n{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    print(f"Train R²={train_score:.4f}, Test R²={test_score:.4f}")

    results.append([name, mae, rmse, r2, train_score, test_score])

# Step 3: Comparison Table
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R²", "Train R²", "Test R²"])
print("\n Regression Model Comparison:")
print(results_df)


# =========================
# Sprint-3: Optimization & Final Model
# =========================

from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

# Step 1: Feature Engineering
df['hp_per_cylinder'] = df['engine_hp'] / df['engine_cylinders'].replace(0, np.nan)
df['log_msrp'] = np.log1p(df['msrp'])
df['mpg_ratio'] = df['highway_mpg'] / df['city_mpg']
print(" Feature engineering complete")

# Step 2: Feature Selection (RFE)
rfe_model = LinearRegression()
rfe = RFE(rfe_model, n_features_to_select=20)
rfe.fit(X_train, y_train)
print(" Feature selection complete")

# Step 3: Hyperparameter Tuning
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best RF params:", grid_search.best_params_)

gb = GradientBoostingRegressor(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
random_search = RandomizedSearchCV(gb, param_dist, n_iter=5, cv=3, scoring='r2', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)
print("Best GB params:", random_search.best_params_)

# Step 4: Final Model
final_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X_train, y_train)

# Step 5: Final Evaluation
y_pred = final_model.predict(X_test)
print("\n--- Final Model Evaluation ---")
print(f"MAE={mean_absolute_error(y_test, y_pred):.2f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.2f}, R²={r2_score(y_test, y_pred):.4f}")

# Step 6: Serialization
joblib.dump(final_model, "final_model_sprint3.pkl")
print(" Final model saved as final_model_sprint3.pkl")
