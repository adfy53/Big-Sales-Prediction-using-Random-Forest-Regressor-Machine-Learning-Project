# Title of Projet
Big Sales Prediction using Random Forest Regressor

# Objective
The objective of this project is to predict sales based on various factors using a Random Forest Regressor.

# Data Source
The dataset used for this project can be sourced from [provide source, e.g., Kaggle, a specific link, etc.].

# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Import Data
# Replace 'file_path' with the actual path to your dataset.
data = pd.read_csv('file_path')
data.head()

# Describe Data
data.info()
data.describe()

# Data Visualization
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(data)
plt.show()

# Data Preprocessing
# Handle missing values
data.fillna(data.mean(), inplace=True)

# Encode categorical variables
# Replace 'categorical_column' with your dataset's categorical column names.
data = pd.get_dummies(data, columns=['categorical_column'], drop_first=True)

# Define Target Variable (y) and Feature Variables (X)
X = data.drop('target_column', axis=1)  # Replace 'target_column' with the name of your target variable.
y = data['target_column']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Prediction
# Replace 'new_data' with the data you want to predict sales for.
new_data = np.array([[value1, value2, value3]])  # Replace with appropriate values.
new_prediction = model.predict(new_data)
print("Predicted Sales:", new_prediction)

# Explanation
# Random Forest Regressor uses multiple decision trees to make predictions, averaging their results for accuracy.
# The model can handle non-linearity and interactions between features, making it a robust choice for this task.
