# """
# Training script for Medical Premium Price Prediction using RandomForestRegressor
# """
#
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# #from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
# import joblib
#
# # Load the medical premium dataset
# data = pd.read_csv("data/Medicalpremium.csv")
#
# # Display basic info (optional)
# print("Dataset preview:")
# print(data.head())
# print("\nTarget distribution:")
# print(data["PremiumPrice"].describe())
#
# # Define features and target
# X = data.drop("PremiumPrice", axis=1)
# y = data["PremiumPrice"]
#
# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Initialize and train the regression model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Predict on test set
# y_pred = model.predict(X_test)
#
# # Evaluate model performance
# r2 = r2_score(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# # Calculate RMSE manually for better compatibility
# mse = mean_squared_error(y_test, y_pred)
# rmse = mse**0.5  # This is the square root
# #rmse = mean_squared_error(y_test, y_pred, squared=False)
# rmse = root_mean_squared_error(y_test, y_pred)
#
# print("\nModel Evaluation Metrics:")
# print(f"R² Score: {r2:.4f}")
# print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
# print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")
#
# # Save the trained model
# joblib.dump(model, "model/model.pkl")
# print("\n✅ Model saved successfully to model/model.pkl")


"""
Training script for Medical Premium Price Prediction using RandomForestRegressor
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib

# 1. Load the medical premium dataset
data = pd.read_csv("data/Medicalpremium.csv")

# 2. Define features and target
X = data.drop("PremiumPrice", axis=1)
y = data["PremiumPrice"]

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and train the regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predict on test set
y_pred = model.predict(X_test)

# 6. Evaluate model performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
# Using the modern Scikit-Learn 1.4+ function
rmse = root_mean_squared_error(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): ₹{mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ₹{rmse:.2f}")

# 7. Save the trained model
# Ensure the 'model' directory exists before saving
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, "model/model.pkl")
print("\n✅ Model saved successfully to model/model.pkl")