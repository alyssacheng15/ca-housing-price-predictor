# Flight Price Prediction Machine Learning Model
Machine learning regression model to precisely predict flight prices using datasets from kaggle and feature engineering. Employs Random Forest regression and fine-tuned with hyperparameter optimization and randomized search, using Python, scikit-learn, pandas, and matplotlib. Forecasts prices based on various features such as airline, source and destination cities, departure and arrival times, class of travel, and flight duration.

## Features
### Data Preprocessing:
#### One-Hot Encoding:
Converts categorical variables such as airline, source city, destination city, departure time, and arrival time into numerical format.
#### Binary Encoding:
Transforms the 'class' feature into binary format, distinguishing between Economy (0) and Business (1).
#### Factorization:
Converts the 'stops' feature into numerical values for ease of model processing.

### Model Training and Evaluation:
#### Training:
Utilizes a Random Forest Regressor to train the model on historical flight data.
#### Evaluation Metrics:
Includes R² score, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess model performance.
#### Feature Importance:
Identifies the most influential features affecting flight prices, with 'class' being the most significant.

### Hyperparameter Tuning:
#### Default Model:
Initial Random Forest Regressor model without parameter tuning.
#### Tuned Model:
Improved model using RandomizedSearchCV to optimize hyperparameters for better accuracy and performance.

## Data
The dataset used includes the following features:
* airline
* flight (removed during preprocessing).
* source city:
* departure time:
* stops (zero, one, or two or more)
* arrival time:
* destination city:
* class (economy or business)
* duration
* days left before flight
* price

## Steps
### Data Preprocessing:
* Load the dataset and perform initial explorations.
* Drop irrelevant columns and encode categorical variables.
* Apply factorization and binarization where necessary.

### Model Training:
* Split the dataset into training and testing sets.
* Train the Random Forest Regressor using the training set.

### Model Evaluation:
* Evaluate the model using metrics such as R², MAE, MSE, and RMSE.
* Generate and analyze feature importance to understand which factors most influence flight prices.

### Hyperparameter Tuning (Optional):
* Use RandomizedSearchCV to find the best hyperparameters for the model.
* Compare the performance of the tuned model with the default model.

## Visualization:
* Create scatter plots to visualize the correlation between actual and predicted flight prices.

## Results
The tuned model shows a slight improvement in R² and error metrics compared to the default model, indicating enhanced prediction accuracy. Overall, the tuned model is pretty accurate in predicting the flight price, as the margin errors are miniscule. Looking at the feature importances, the most important feature is by far class, followed by duration and days left.
