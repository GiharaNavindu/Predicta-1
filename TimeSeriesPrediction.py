import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


historical_data = pd.read_csv(r"historical_weather.csv")
submission_key = pd.read_csv(r"submission_key.csv")
sample_submission = pd.read_csv(r"sample_submission.csv")
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data['month'] = historical_data['date'].dt.month
historical_data['day'] = historical_data['date'].dt.day


# visualizations


# Plot the distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(historical_data['avg_temp_c'].dropna(), kde=True, bins=30)
plt.title('Distribution of Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Frequency')
plt.show()


# Plot missing values for each dataset

def plot_missing_values(df, title):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)

    plt.figure(figsize=(10, 6))
    missing.plot.bar()
    plt.title(f'Missing Values in {title}')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.show()


# Plot missing values for each dataset
plot_missing_values(historical_data, 'Historical Data')

# Preprocess the data
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data['month'] = historical_data['date'].dt.month
historical_data['day'] = historical_data['date'].dt.day

# Encode city_id
le = LabelEncoder()
historical_data['city_id_encoded'] = le.fit_transform(historical_data['city_id'])

# Identify numeric and categorical columns
numeric_features = ['min_temp_c', 'max_temp_c', 'precipitation_mm', 'snow_depth_mm', 'avg_wind_dir_deg',
                    'avg_wind_speed_kmh']
categorical_features = ['city_id_encoded', 'month', 'day']

# Prepare features and target
features = categorical_features + numeric_features
X = historical_data[features]
y = historical_data['avg_temp_c']

# Drop NaN values from y and corresponding rows from X
mask = ~y.isna() & ~X.isna().any(axis=1)
X = X[mask]
y = y[mask]

print(f"Number of samples after dropping NaN target values: {len(y)}")

# Impute missing values in X
# For numeric features, use Iterative Imputer (also known as MICE)
numeric_imputer = IterativeImputer(random_state=42)
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# For categorical features, use mode imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=2500, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
val_predictions = model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
print(f"Validation RMSE: {rmse}")

# Prepare submission data
submission_key['date'] = pd.to_datetime(submission_key['date'])
submission_key['month'] = submission_key['date'].dt.month
submission_key['day'] = submission_key['date'].dt.day
submission_key['city_id_encoded'] = le.transform(submission_key['city_id'])

# Merge with historical data to get other features
last_day_data = historical_data.groupby('city_id').last().reset_index()
submission_data = submission_key.merge(last_day_data[['city_id'] + numeric_features], on='city_id', how='left')

# Impute any missing values in submission data
submission_data[numeric_features] = numeric_imputer.transform(submission_data[numeric_features])
submission_data[categorical_features] = cat_imputer.transform(submission_data[categorical_features])

# Make predictions
X_submit = submission_data[features]
predictions = model.predict(X_submit)






# Prepare submission file
sample_submission['avg_temp_c'] = predictions
sample_submission.to_csv('sample_submission.csv', index=False)

print("Submission file created: sample_submission.csv")

#Encoding
le = LabelEncoder()
historical_data['city_id_encoded'] = le.fit_transform(historical_data['city_id'])

#numeric and categorical columns
numeric_features = ['min_temp_c', 'max_temp_c', 'precipitation_mm', 'snow_depth_mm', 'avg_wind_dir_deg', 'avg_wind_speed_kmh']
categorical_features = ['city_id_encoded', 'month', 'day']

#features and target
features = categorical_features + numeric_features
X = historical_data[features]
y = historical_data['avg_temp_c']

# Drop NaN values 
mask = ~y.isna() & ~X.isna().any(axis=1)
X = X[mask]
y = y[mask]
print(f"Number of samples after dropping NaN target values: {len(y)}")

# Impute missing values in X
# For numeric features, use Iterative Imputer (also known as MICE)
numeric_imputer = IterativeImputer(random_state=42)
X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])

# For categorical features, use mode imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#Random Forest model  training
model = RandomForestRegressor(n_estimators=2500, random_state=42)
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
rmse = np.sqrt(mse)
print(f"Validation RMSE: {rmse}")#RMSE


#submission 
submission_key['date'] = pd.to_datetime(submission_key['date'])
submission_key['month'] = submission_key['date'].dt.month
submission_key['day'] = submission_key['date'].dt.day
submission_key['city_id_encoded'] = le.transform(submission_key['city_id'])
last_day_data = historical_data.groupby('city_id').last().reset_index()
submission_data = submission_key.merge(last_day_data[['city_id'] + numeric_features], on='city_id', how='left')

# Impute any missing values in submission data
submission_data[numeric_features] = numeric_imputer.transform(submission_data[numeric_features])
submission_data[categorical_features] = cat_imputer.transform(submission_data[categorical_features])

#predictions
X_submit = submission_data[features]
predictions = model.predict(X_submit)



# Prepare submission file
sample_submission['avg_temp_c'] = predictions
sample_submission.to_csv('sample_submission.csv', index=False)
print("Submission file created: sample_submission.csv")
