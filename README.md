# Car Price Prediction using Machine Learning

Predicting car prices accurately is crucial for both buyers and sellers. This project utilizes machine learning techniques to forecast car prices with precision, facilitating informed decisions in the automotive market.

## Table of Contents
- [Introduction](#introduction)
- [Steps](#steps)
- [Architecture](#architecture)
- [Training](#training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Contact](#contact)
- [References](#references)
- [Project Link](#project-link)

## Introduction
This project harnesses the power of machine learning, particularly the XGBRegressor algorithm, to predict car prices accurately. By incorporating advanced feature engineering and meticulous model tuning, it achieves optimal performance, providing invaluable insights into the automotive market.

## Steps

1. **Data Preparation**
   - Imported necessary libraries.
   - Read the dataset using Pandas.
   - Explored the structure and contents of the dataset.

2. **Data Preprocessing**
   - Performed label encoding for categorical features.
   - Visualized categorical feature distributions using count plots.

3. **Feature Engineering**
   - Created a new feature 'Year_Mileage' by multiplying 'year' and 'mileage'.
   - Visualized the relationship between 'Year_Mileage' and 'Price' using scatter plots.

4. **Data Scaling**
   - Scaled numerical features using StandardScaler.
   - Visualized feature distributions before and after scaling.

5. **Model Training**
   - Split the dataset into training and testing sets.
   - Trained the XGBRegressor model without feature engineering.
   - Evaluated the model's performance using R2 score.

6. **Hyperparameter Tuning**
   - Fine-tuned model hyperparameters using RandomizedSearchCV.
   - Selected the best model based on performance metrics.
   - Evaluated the best model's performance on the test set.

7. **Model Evaluation**
   - Calculated Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
   - Visualized residuals and actual vs. predicted prices using scatter plots and residual plots.

8. **GridSearchCV**
   - Performed GridSearchCV to further optimize model hyperparameters.
   - Evaluated the best model's performance and compared it with RandomizedSearchCV.

9. **Feature Importance**
   - Determined feature importances using the best model.
   - Visualized feature importances to identify the most influential features.

10. **Top Feature Analysis**
    - Selected the top 5 features based on importance.
    - Trained a new model using only the top 5 features.
    - Evaluated the performance of the model with the selected features.

These steps outline the comprehensive approach followed to predict car prices, showcasing the integration of various techniques in machine learning for accurate valuation.

## Architecture
At the core of the model lies the XGBRegressor algorithm, strategically enhanced with the 'Year_Mileage' attribute through feature engineering. This augmentation enables the model to capture nuanced correlations between car age and mileage, resulting in precise price predictions.

## Training
The dataset undergoes meticulous preprocessing, encompassing label encoding and Standard Scaler methods, ensuring robust model training and improved performance.

## Hyperparameter Tuning
Leveraging RandomizedSearchCV, the model's hyperparameters undergo fine-tuning, achieving an impressive R2 score of 96.7% within a training duration of 11.06 seconds.

## Evaluation
The model's efficacy is demonstrated by its Mean Absolute Error (MAE) of $1487.30, affirming its accuracy in predicting car prices, thereby serving as a valuable asset for pricing analysis.

## Features
- **Feature Engineering**: Introduction of the 'Year_Mileage' attribute.
- **Data Preprocessing**: Utilization of label encoding and Standard Scaler.
- **Hyperparameter Tuning**: Optimization using RandomizedSearchCV.
- **High Accuracy**: Attainment of an R2 score of 96.7%.
- **Efficiency**: Completion of model training in 11.06 seconds.
- **Performance Metric**: MAE of $1487.30.

## Technologies Used
- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy

## Contact
For inquiries or feedback, feel free to reach out:
- [Gmail](mailto:mr.muadrahman@gmail.com)
- [LinkedIn](https://www.linkedin.com/in/muadrahman/)

## References
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## Project Link
For further details and access to the project repository, visit [this link](https://github.com/muadrahman/Car-Price-Prediction).

