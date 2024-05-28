#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time


# In[2]:


# Read the dataset
file_path = r"D:\CV things\ML projects\audi.csv"
df = pd.read_csv(file_path)
print(df.shape)
df


# In[3]:


# Label Encoding for categorical features
categorical_features = ['model', 'transmission', 'fuelType']
le = LabelEncoder()
df[categorical_features] = df[categorical_features].apply(lambda col: le.fit_transform(col.astype(str)))
df.head()


# In[4]:


# Countplot for categorical variables
plt.figure(figsize=(10, 6))
for col in categorical_features:
    sns.countplot(data=df, x=col)
    plt.title(f'Countplot of {col}')
    plt.show()


# In[5]:


# Preprocessing
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# In[6]:


# R2 Score before Feature Engineering
xgb = XGBRegressor()
 
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
r2_base = r2_score(y_test, y_pred)
print(f"R2 Score before feature engineering: {r2_base}")


# In[7]:


# Feature Engineering - Creating a new feature 'Year_Mileage'
df['Year_Mileage'] = df['year'] * df['mileage']
df.head()


# In[8]:


# Scatterplot showing Year_Mileage vs. Price after feature engineering
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Year_Mileage', y='price', alpha=0.5, palette='viridis')
plt.title('Year_Mileage vs. Price')
plt.xlabel('Year_Mileage')
plt.ylabel('Price')
plt.show()


# In[9]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[10]:


# Preprocessing
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# R2 Score after Feature Engineering
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
r2_base = r2_score(y_test, y_pred)

print(f"R2 Score after feature engineering: {r2_base}")


# In[11]:


# Get numerical columns and set the ones to display
numerical_columns = df.select_dtypes(include='number').columns.tolist()
num_cols_to_display = 9
num_cols = numerical_columns[:num_cols_to_display]

# Visualize distributions before scaling for each numerical feature separately
plt.figure(figsize=(15, 15))

for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col], bins=20, kde=True, color='skyblue')
    plt.title(f"Distribution of {col} (Before Scaling)")
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()


# In[12]:


# Assuming df contains your dataset and numerical_columns are the columns you want to scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numerical_columns])

# Create a DataFrame for scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_columns)
X_scaled_df['price'] = df['price']  # Include the target variable if needed

X = X_scaled_df.drop('price', axis=1)
y = X_scaled_df['price']

X_scaled_df.head()


# In[13]:


# Visualize distributions after scaling for each numerical feature separately
plt.figure(figsize=(15, 15))

for i, col in enumerate(num_cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(X_scaled_df[col], bins=20, kde=True, color='salmon')  # Adjust color as needed
    plt.title(f"Distribution of {col} (After Scaling)")
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()


# In[14]:


# Assuming you've performed the train-test split previously
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, test_size=0.2, random_state=40)


# In[15]:


# R2 Score after scaling
scaled_xgb = XGBRegressor()
scaled_xgb.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = scaled_xgb.predict(X_test_scaled)
r2_scaled = r2_score(y_test_scaled, y_pred_scaled)
print(f"R2 Score after scaling: {r2_scaled}")


# In[16]:


# Hyperparameter tuning
param_dist = {
    'n_estimators': range(100, 300),
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': range(3, 8)
}

# Initialize XGBoost regressor and RandomizedSearchCV
xgb = XGBRegressor()

# RandomizedSearchCV
start_time_random = time.time()

random_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist, n_iter=20, cv=5, scoring='r2')

# Perform RandomizedSearchCV on the scaled training data
random_search.fit(X_train_scaled, y_train_scaled)

# Get the best estimator
best_xgb_random = random_search.best_estimator_

end_time_random = time.time()
random_search_time = end_time_random - start_time_random

# Predictions using the best model
y_pred_random = best_xgb_random.predict(X_test_scaled)


# In[17]:


# Print mean scores and standard deviations for different hyperparameter combinations
cv_results = random_search.cv_results_
for mean_score, std_score, params in zip(
    cv_results["mean_test_score"],
    cv_results["std_test_score"],
    cv_results["params"]
):
    print(f"Mean R2: {mean_score:.4f}, Std: {std_score:.4f} for {params}")


# In[18]:


# Evaluate the model
r2_random = r2_score(y_test_scaled, y_pred_random)
mae = mean_absolute_error(y_test_scaled, y_pred_random)
mse = mean_squared_error(y_test_scaled, y_pred_random)
rmse = np.sqrt(mse)

print("Best XGBoost Model Parameters:", random_search.best_params_)


# In[19]:


print(f"R2 Score after RandomizedSearchCV: {r2_random}")

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# In[20]:


# Residual plot
residuals = y_test_scaled - y_pred_random
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_scaled, y=residuals, lowess=True, color='skyblue')
plt.title('Residual Plot')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()


# In[21]:


# Scatterplot for actual vs. predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_scaled, y=y_pred_random)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# In[22]:


# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7]
}

# Initialize XGBoost regressor
xgb = XGBRegressor()

# GridSearchCV
start_time_grid = time.time()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2')

# Perform GridSearchCV on the scaled training data
grid_search.fit(X_train_scaled, y_train_scaled)

end_time_grid = time.time()
grid_search_time = end_time_grid - start_time_grid

# Get the best estimator
best_xgb_grid = grid_search.best_estimator_

# Predictions using the best model from GridSearchCV
y_pred_grid = best_xgb_grid.predict(X_test_scaled)


# In[23]:


# Print mean scores and standard deviations for different hyperparameter combinations
cv_results = grid_search.cv_results_
for mean_score, std_score, params in zip(
    cv_results["mean_test_score"],
    cv_results["std_test_score"],
    cv_results["params"]
):
    print(f"Mean R2: {mean_score:.4f}, Std: {std_score:.4f} for {params}")


# In[24]:


# Evaluate the model
r2_grid = r2_score(y_test_scaled, y_pred_grid)

print("Best XGBoost Model Parameters (GridSearchCV):", grid_search.best_params_)
print(f"R2 Score after GridSearchCV: {r2_grid}")


# In[25]:


# Residual plot
residuals = y_test_scaled - y_pred_grid
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred_grid, y=residuals, lowess=True, color='skyblue')
plt.title('Residual Plot')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.show()


# In[26]:


# Scatterplot for actual vs. predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_scaled, y=y_pred_grid)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()


# In[27]:


print(f"RandomizedSearchCV took {random_search_time} seconds.")
print(f"GridSearchCV took {grid_search_time} seconds.")


# In[28]:


# Display feature importance for the best model in decreasing order and different colors
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(best_xgb_grid.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)  # Sort in decreasing order
sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')


# In[29]:


# Select the top 5 features based on importance
top_features = feat_importances.index[:5]

# Extract only the top 5 features from the original dataset
X_top5 = X[top_features]

# Train-test split with the selected features
X_train_top5, X_test_top5, y_train_top5, y_test_top5 = train_test_split(X_top5, y, test_size=0.2, random_state=40)

# Initialize XGBoost regressor
xgb_top5 = XGBRegressor()

# Fit the model using the top 5 features
xgb_top5.fit(X_train_top5, y_train_top5)

# Make predictions
y_pred_top5 = xgb_top5.predict(X_test_top5)

# Evaluate the model with the top 5 features
r2_top5 = r2_score(y_test_top5, y_pred_top5)

print(f"R2 Score with the top 5 features: {r2_top5}")

