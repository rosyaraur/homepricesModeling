# Python analysis project: California Housing Price Prediction

This project focuses on predicting median house values in California using the California Housing Dataset, derived from the 1990 U.S. Census. The dataset includes features such as median income, house age, average rooms, population, and geographic coordinates. The goal is to build and evaluate multiple regression models to estimate housing prices accurately.

<b> Methodology Overview </b>

1. Data Preparation
Loaded the dataset using fetch_california_housing from sklearn.

Split into features (X) and target (y), where the target is MedHouseVal (median house value).

Applied StandardScaler to normalize features for models sensitive to scale.

2. Model Selection
   
A diverse set of regression models were selected:

Traditional Models: SVR, RandomForestRegressor, GradientBoostingRegressor

Advanced Models: XGBoost, LightGBM, CatBoost, ElasticNet, MLPRegressor

Ensemble Techniques:

Weighted Ensemble: Averaged predictions with predefined weights.

Stacking Ensemble: Combined base model outputs using a meta-model (LinearRegression).

3. Model Evaluation
   
Used RepeatedKFold cross-validation with 1000 replications to assess model stability.

Calculated R² scores to measure predictive accuracy.

Visualized:

Boxplots of R² distributions

Scatter plots of actual vs predicted values

Residual plots to diagnose bias and variance

4. Simulation and Prediction
   
Simulated a new dataset of 10 hypothetical homes with randomized feature values.

Trained models on the full dataset and predicted house values for these new homes.

Compared predictions across individual models, weighted ensemble, and stacking ensemble.

5. Interactive Dashboard
   
Built a Streamlit dashboard allowing users to:

Input property features via sliders

Select models to use for prediction

View predicted house values in real-time

6. Deployment
   
Prepared the dashboard for deployment using Streamlit Community Cloud.

Included a requirements.txt file for dependency management.

Enabled public access via GitHub integration and cloud hosting.

<b> Outcome </b>
The stacking ensemble and advanced gradient boosting models (XGBoost, LightGBM, CatBoost) demonstrated superior performance in terms of accuracy and consistency. The dashboard provides an intuitive interface for real-time prediction, making the model accessible for practical use.
