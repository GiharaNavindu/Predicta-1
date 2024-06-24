import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


data = pd.read_csv(r"daily_data.csv")
submission_template = pd.read_csv(r"submission.csv")

# Preprocessing
def preprocess_data(df):
    def time_to_minutes(time_str):
        if pd.isna(time_str):
            return np.nan
        parts = time_str.split()
        if len(parts) == 2:
            time, am_pm = parts
            hour, minute = map(int, time.split(':'))
            if am_pm == 'PM' and hour != 12:
                hour += 12
            elif am_pm == 'AM' and hour == 12:
                hour = 0
        else:
            hour, minute = map(int, time_str.split(':'))
        return hour * 60 + minute

    df['sunrise'] = df['sunrise'].apply(time_to_minutes)
    df['sunset'] = df['sunset'].apply(time_to_minutes)
    
    # Feature engineering
    df['day_length'] = df['sunset'] - df['sunrise']
    df['temp_humidity_ratio'] = df['temperature_celsius'] / df['humidity']
    df['wind_chill'] = 13.12 + 0.6215 * df['temperature_celsius'] - 11.37 * (df['wind_kph'] ** 0.16) + 0.3965 * df['temperature_celsius'] * (df['wind_kph'] ** 0.16)
    df['heat_index'] = -8.78469475556 + 1.61139411 * df['temperature_celsius'] + 2.33854883889 * df['humidity'] - 0.14611605 * df['temperature_celsius'] * df['humidity'] - 0.012308094 * df['temperature_celsius']*2 - 0.0164248277778 * df['humidity']2 + 0.002211732 * df['temperature_celsius']2 * df['humidity'] + 0.00072546 * df['temperature_celsius'] * df['humidity']2 - 0.000003582 * df['temperature_celsius']2 * df['humidity']*2
    
    return df

data = preprocess_data(data)


X = data.drop(['day_id', 'condition_text'], axis=1)
y = data['condition_text']
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])






preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# targeet
le = LabelEncoder()
y_encoded = le.fit_transform(y.dropna())  


X_train_full = X[y.notna()]
y_train_full = y_encoded
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

#models
rf = RandomForestClassifier(n_estimators=200, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)

#Voting classifier
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'
)

# Create a pipeline with the preprocessor and the voting classifier
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', voting_clf)])

# Define hyperparameters for RandomizedSearchCV
param_dist = {
    'classifier_rf_max_depth': [10, 20, 30, None],
    'classifier_rf_min_samples_split': [2, 5, 10],
    'classifier_rf_min_samples_leaf': [1, 2, 4],
    'classifier_gb_learning_rate': [0.01, 0.1, 0.2],
    'classifier_gb_max_depth': [3, 5, 7],
    'classifier_gb_min_samples_split': [2, 5, 10],
    'classifier_gb_min_samples_leaf': [1, 2, 4]
}

#RandomizedSearchCV
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

#Checking the accuracy
y_val_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Model Accuracy: {accuracy:.4f}")

#traing again
best_model.fit(X_train_full, y_train_full)


submission_data = data[data['condition_text'].isna()]
submission_features = submission_data.drop(['day_id', 'condition_text'], axis=1)
submission_pred = best_model.predict(submission_features)
submission_pred_labels = le.inverse_transform(submission_pred)
submission = pd.DataFrame({
    'day_id': submission_data['day_id'],
    'condition_text': submission_pred_labels
})

#final submission
final_submission = pd.concat([
    data[data['condition_text'].notna()][['day_id', 'condition_text']],
    submission
]).sort_values('day_id').reset_index(drop=True)
final_submission.to_csv('submission.csv', index=False)
print("Submission file created successfully: submission.csv")