
# 1. California Housing Dataset
# California Housing Dataset Variables
##The dataset contains 20,640 instances and 8 features plus the target.
##All features are numeric and predictive.
##The target variable (MedHouseVal) is continuous, making this a regression problem.
##Variable	Description
##MedInc	        Median income in block group (in tens of thousands of dollars)
##HouseAge	Median age of houses in the block group
##AveRooms	Average number of rooms per household
##AveBedrms	Average number of bedrooms per household
##Population	Total population of the block group
##AveOccup	Average number of occupants per household
##Latitude	Geographical latitude of the block group
##Longitude	Geographical longitude of the block group
##MedHouseVal	Target variable: Median house value (in hundreds of thousands of dollars)

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
print(df.head())

# build a regression model to predict median house value (MedHouseVal) based on the other features.
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Split Features and Target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the Model
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize Predictions
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted House Values")
plt.show()

# use SVR, RandomForestRegressor, and GradientBoostingRegressor on the California Housing Dataset,
# with StandardScaler for normalization and cross-validation for robust evaluation
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Define models with scaling
models = {
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1)),
    'RandomForest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
    'GradientBoosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
}

# Evaluate each model using 5-fold cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name} R² scores: {np.round(scores, 3)}")
    print(f"{name} Mean R²: {scores.mean():.3f}\n")

# StandardScaler ensures features are normalized before modeling.
# SVR uses a radial basis function kernel for nonlinear regression.
# RandomForest and GradientBoosting are ensemble methods that often outperform linear models.
# Cross-validation gives a more reliable estimate of model performance.

# Visualization of Predictions

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1)),
    'RandomForest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
    'GradientBoosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
}

# Plot predictions
plt.figure(figsize=(18, 5))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.subplot(1, 3, i)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal line
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name} Predictions")

plt.tight_layout()
plt.show()

# tune hyperparameters for better accuracy
# Let’s take your modeling to the next level by tuning hyperparameters for SVR, RandomForestRegressor, and
# GradientBoostingRegressor using GridSearchCV. This will help you find the best combination of parameters
# for each model and improve accuracy.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grids
param_grids = {
    'SVR': {
        'svr__C': [10, 100],
        'svr__gamma': [0.01, 0.1],
        'svr__kernel': ['rbf']
    },
    'RandomForest': {
        'randomforestregressor__n_estimators': [100, 200],
        'randomforestregressor__max_depth': [None, 10, 20]
    },
    'GradientBoosting': {
        'gradientboostingregressor__n_estimators': [100, 200],
        'gradientboostingregressor__learning_rate': [0.05, 0.1],
        'gradientboostingregressor__max_depth': [3, 5]
    }
}

# Define models
models = {
    'SVR': make_pipeline(StandardScaler(), SVR()),
    'RandomForest': make_pipeline(StandardScaler(), RandomForestRegressor(random_state=42)),
    'GradientBoosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=42))
}

# Perform GridSearchCV
best_models = {}
for name in models:
    print(f"🔍 Tuning {name}...")
    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"✅ Best params for {name}: {grid.best_params_}")
    print(f"📈 Best R² score: {grid.best_score_:.3f}\n")
    best_models[name] = grid.best_estimator_

# Uses GridSearchCV to explore combinations of hyperparameters.
# Applies cross-validation for robust scoring.
# Returns the best model configuration for each algorithm.

# 1000 replications of cross-validation
# Running 1000 replications of cross-validation is a robust way to assess model stability and performance.
# Here's how you can do it using RepeatedKFold from sklearn, which performs multiple rounds of K-Fold
# cross-validation with different splits each time.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Define best models based on previous tuning
models = {
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1)),
    'RandomForest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    'GradientBoosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
}

# Define 1000-replication cross-validation strategy
cv = RepeatedKFold(n_splits=5, n_repeats=200, random_state=42)

# Evaluate each model
for name, model in models.items():
    print(f"🔁 Evaluating {name} with 1000-fold CV...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    print(f"✅ Mean R²: {scores.mean():.4f}")
    print(f"📉 Std Dev R²: {scores.std():.4f}")
    print(f"📊 Min R²: {scores.min():.4f}")
    print(f"📈 Max R²: {scores.max():.4f}\n")

# Mean R²: Average accuracy across all 1000 folds.
# Standard deviation: Measures consistency.
# Min/Max R²: Shows worst and best-case performance.
# This gives you a rock-solid understanding of how each model performs under varied data splits.

# Boxplot of R² Scores
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Define models with best configurations
models = {
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1)),
    'RandomForest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    'GradientBoosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))
}

# Define 1000-replication cross-validation strategy
cv = RepeatedKFold(n_splits=5, n_repeats=200, random_state=42)

# Collect R² scores
results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
    for score in scores:
        results.append({'Model': name, 'R2 Score': score})

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Plot boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Model', y='R2 Score', data=df_results, palette='Set2')
plt.title('Distribution of R² Scores Across 1000 Replications')
plt.ylabel('R² Score')
plt.xlabel('Model')
plt.grid(True)
plt.show()

# Boxplots show the spread, median, and outliers of R² scores for each model.
# Helps you compare accuracy and stability.
# A narrower box means more consistent performance; a higher median means better average accuracy.

# Pediction for new homes
# Let's simulate a new dataset of 10 homes, build an ensemble model using SVR, RandomForestRegressor, and GradientBoostingRegressor,
# and then use it to predict the median house value for those homes.
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

# Load original dataset for feature reference
data = fetch_california_housing(as_frame=True)
feature_names = data.feature_names

# Simulate new dataset of 10 homes
np.random.seed(42)
new_homes = pd.DataFrame({
    'MedInc': np.random.uniform(1, 15, 10),          # Median income
    'HouseAge': np.random.uniform(1, 50, 10),        # House age
    'AveRooms': np.random.uniform(2, 10, 10),        # Average rooms
    'AveBedrms': np.random.uniform(1, 5, 10),        # Average bedrooms
    'Population': np.random.uniform(100, 5000, 10),  # Population
    'AveOccup': np.random.uniform(1, 5, 10),         # Average occupancy
    'Latitude': np.random.uniform(32, 42, 10),       # Latitude in CA
    'Longitude': np.random.uniform(-124, -114, 10)   # Longitude in CA
}, columns=feature_names)

# Define models with best configurations
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1))
rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
gb = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))

# Train models on full original dataset
X = data.data
y = data.target
svr.fit(X, y)
rf.fit(X, y)
gb.fit(X, y)

# Predict with each model
svr_preds = svr.predict(new_homes)
rf_preds = rf.predict(new_homes)
gb_preds = gb.predict(new_homes)

# Ensemble prediction: average of all models
ensemble_preds = (svr_preds + rf_preds + gb_preds) / 3

# Show predictions
results = new_homes.copy()
results['SVR_Pred'] = svr_preds
results['RF_Pred'] = rf_preds
results['GB_Pred'] = gb_preds
results['Ensemble_Pred'] = ensemble_preds

print(results[['SVR_Pred', 'RF_Pred', 'GB_Pred', 'Ensemble_Pred']])

# Simulates realistic home features for 10 properties.
# Trains each model on the full California Housing dataset.
# Predicts house values using each model.
# Combines predictions using a simple average ensemble.


# Weighted Ensemble Prediction
# We'll use the same simulated 10-home dataset and apply weights based on assumed model performance (e.g., R² scores from previous evaluations)

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

# Load original dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
feature_names = data.feature_names

# Simulate new dataset of 10 homes
np.random.seed(42)
new_homes = pd.DataFrame({
    'MedInc': np.random.uniform(1, 15, 10),
    'HouseAge': np.random.uniform(1, 50, 10),
    'AveRooms': np.random.uniform(2, 10, 10),
    'AveBedrms': np.random.uniform(1, 5, 10),
    'Population': np.random.uniform(100, 5000, 10),
    'AveOccup': np.random.uniform(1, 5, 10),
    'Latitude': np.random.uniform(32, 42, 10),
    'Longitude': np.random.uniform(-124, -114, 10)
}, columns=feature_names)

# Define models with best configurations
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1))
rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
gb = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))

# Train models
svr.fit(X, y)
rf.fit(X, y)
gb.fit(X, y)

# Predict
svr_preds = svr.predict(new_homes)
rf_preds = rf.predict(new_homes)
gb_preds = gb.predict(new_homes)

# Define weights (based on prior R² performance, for example)
weights = {
    'SVR': 0.2,
    'RandomForest': 0.4,
    'GradientBoosting': 0.4
}

# Weighted ensemble prediction
ensemble_preds = (
    weights['SVR'] * svr_preds +
    weights['RandomForest'] * rf_preds +
    weights['GradientBoosting'] * gb_preds
)

# Display results
results = new_homes.copy()
results['SVR_Pred'] = svr_preds
results['RF_Pred'] = rf_preds
results['GB_Pred'] = gb_preds
results['Weighted_Ensemble_Pred'] = ensemble_preds

print(results[['SVR_Pred', 'RF_Pred', 'GB_Pred', 'Weighted_Ensemble_Pred']])

# Gives more influence to models that perform better.
# Reduces the impact of weaker models.
# Often improves overall prediction accuracy.

# stacking ensemble model
# a powerful technique where predictions from multiple base models are used as inputs to a meta-model that
# learns how to best combine them. This often boosts accuracy by leveraging the strengths of each model.
# Use SVR, RandomForestRegressor, and GradientBoostingRegressor as base models.
# Use LinearRegression as the meta-model.
# Apply StandardScaler to normalize features.
# Use StackingRegressor from sklearn.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('svr', make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1))),
    ('rf', make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))),
    ('gb', make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)))
]

# Define meta-model
meta_model = LinearRegression()

# Create stacking ensemble
stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

# Train stacking model
stacked_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = stacked_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"📈 Stacking Ensemble R² Score: {r2:.4f}")

# Base models capture different patterns in the data.
# Meta-model learns to weigh and combine their predictions.
# Cross-validation ensures robust training of the meta-model.

# Compare Model Performance
# Let’s bring the comparison to life with a visualization that shows how the stacking model, weighted ensemble,
# and individual models (SVR, Random Forest, Gradient Boosting) perform on the same test set.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Assume models are already trained and predictions are made on X_test
# y_test is the actual target values

# Predict with individual models
svr_pred = svr.predict(X_test)
rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

# Weighted ensemble prediction
weights = {'SVR': 0.2, 'RandomForest': 0.4, 'GradientBoosting': 0.4}
weighted_pred = (
    weights['SVR'] * svr_pred +
    weights['RandomForest'] * rf_pred +
    weights['GradientBoosting'] * gb_pred
)

# Stacking model prediction
stacked_pred = stacked_model.predict(X_test)

# Calculate R² scores
scores = {
    'SVR': r2_score(y_test, svr_pred),
    'RandomForest': r2_score(y_test, rf_pred),
    'GradientBoosting': r2_score(y_test, gb_pred),
    'Weighted Ensemble': r2_score(y_test, weighted_pred),
    'Stacking Ensemble': r2_score(y_test, stacked_pred)
}

# Bar chart of R² scores
plt.figure(figsize=(10, 6))
plt.bar(scores.keys(), scores.values(), color=['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'])
plt.ylabel('R² Score')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Scatter Plot of Predictions vs Actual
plt.figure(figsize=(12, 6))
for i, (name, pred) in enumerate([
    ('SVR', svr_pred),
    ('RandomForest', rf_pred),
    ('GradientBoosting', gb_pred),
    ('Weighted Ensemble', weighted_pred),
    ('Stacking Ensemble', stacked_pred)
], 1):
    plt.subplot(2, 3, i)
    plt.scatter(y_test, pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(name)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# Which model has the highest R² score.
# How closely each model’s predictions align with actual values.
# Whether stacking or weighted ensemble offers a meaningful boost.

# Residual Plots for All Models
# Residual plots are a great way to diagnose model performance

import matplotlib.pyplot as plt
import numpy as np

# Calculate residuals
svr_resid = y_test - svr_pred
rf_resid = y_test - rf_pred
gb_resid = y_test - gb_pred
weighted_resid = y_test - weighted_pred
stacked_resid = y_test - stacked_pred

# Plot residuals
plt.figure(figsize=(15, 10))
for i, (name, resid) in enumerate([
    ('SVR', svr_resid),
    ('RandomForest', rf_resid),
    ('GradientBoosting', gb_resid),
    ('Weighted Ensemble', weighted_resid),
    ('Stacking Ensemble', stacked_resid)
], 1):
    plt.subplot(3, 2, i)
    plt.scatter(y_test, resid, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{name} Residuals')
    plt.xlabel('Actual Median House Value')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.show()

# Random scatter around zero: Good sign—model is unbiased.
# Patterns or curves: Indicates model is missing structure.
# Funnel shape: Suggests heteroscedasticity (variance changes with value).
# Outliers: Large residuals that may skew performance.

# Full Model Benchmark
# Mode models - we can try
# Previously Evaluated: SVR RandomForestRegressor GradientBoostingRegressor Weighted Ensemble Stacking Ensemble
# New Models: XGBoost LightGBM CatBoost ElasticNet MLPRegressor (Neural Network)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1))
rf = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))
gb = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42))

# Weighted ensemble
weights = {'SVR': 0.2, 'RF': 0.4, 'GB': 0.4}
svr.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
weighted_pred = (
    weights['SVR'] * svr.predict(X_test) +
    weights['RF'] * rf.predict(X_test) +
    weights['GB'] * gb.predict(X_test)
)

# Stacking ensemble
stacked = StackingRegressor(
    estimators=[('svr', svr), ('rf', rf), ('gb', gb)],
    final_estimator=LinearRegression(),
    cv=5, n_jobs=-1
)
stacked.fit(X_train, y_train)

# Advanced models
xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
catboost = CatBoostRegressor(verbose=0, random_state=42)
elastic = make_pipeline(StandardScaler(), ElasticNet(alpha=1.0, l1_ratio=0.5))
mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))

# Train and evaluate all models
models = {
    'SVR': svr,
    'RandomForest': rf,
    'GradientBoosting': gb,
    'Weighted Ensemble': None,  # already predicted
    'Stacking Ensemble': stacked,
    'XGBoost': xgb,
    'LightGBM': lgbm,
    'CatBoost': catboost,
    'ElasticNet': elastic,
    'MLPRegressor': mlp
}

from sklearn.metrics import r2_score

r2_scores = {}
for name, model in models.items():
    if name == 'Weighted Ensemble':
        r2_scores[name] = r2_score(y_test, weighted_pred)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2_scores[name] = r2_score(y_test, pred)

# Plot R² scores
plt.figure(figsize=(12, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color='teal')
plt.ylabel('R² Score')
plt.title('Model Comparison on California Housing Dataset')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Which model performs best on unseen data.
# How advanced models like XGBoost, LightGBM, and CatBoost stack up against traditional and ensemble methods.
# Whether neural networks or regularized linear models offer competitive accuracy.

# Streamlit Dashboard
# interactive dashboard built using Streamlit, a Python library that lets you create web apps for data science with minimal code.
# This dashboard allows users to:
# Input property features (like income, rooms, location, etc.)
# Select one or more models (SVR, Random Forest, Gradient Boosting, etc.)
# Get predicted house values from the selected models
# Save this as app.py and run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.datasets import fetch_california_housing

# Load training data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train models
models = {
    'SVR': make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1)),
    'Random Forest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    'Gradient Boosting': make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
    'XGBoost': XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
    'Stacking Ensemble': StackingRegressor(
        estimators=[
            ('svr', make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma=0.1))),
            ('rf', make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42))),
            ('gb', make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)))
        ],
        final_estimator=LinearRegression(),
        cv=5,
        n_jobs=-1
    )
}

for model in models.values():
    model.fit(X, y)

# Streamlit UI
st.title("🏡 California Housing Price Predictor")
st.write("Enter property details and select models to predict median house value.")

# Input fields
med_inc = st.slider("Median Income (in $10,000s)", 1.0, 15.0, 5.0)
house_age = st.slider("House Age", 1, 50, 20)
ave_rooms = st.slider("Average Rooms", 2.0, 10.0, 5.0)
ave_bedrms = st.slider("Average Bedrooms", 1.0, 5.0, 2.0)
population = st.slider("Population", 100, 5000, 1000)
ave_occup = st.slider("Average Occupancy", 1.0, 5.0, 2.5)
latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# Model selection
selected_models = st.multiselect("Select Models", list(models.keys()), default=['Random Forest', 'Gradient Boosting'])

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([{
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }])

    st.subheader("📈 Predicted Median House Values")
    for name in selected_models:
        prediction = models[name].predict(input_data)[0]
        st.write(f"**{name}**: ${prediction * 100_000:,.2f}")

# Install Streamlit: pip install streamlit
# Save the code above as app.py
# Run: streamlit run app.py
# Use the sliders and dropdown to interact with the models



