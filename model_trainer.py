
import pandas as pd
import numpy as np
import pickle

# sklearn preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# regression
from sklearn.ensemble import RandomForestRegressor

# metrices
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import warnings
warnings.filterwarnings("ignore")

# load dataset

df = pd.read_csv("vgsales.csv")


# Categorical and Numerical features and Target
target_col = "Global_Sales"
df = df.drop(columns=["Name"])


# Preprocessing

# Handle Missing Values

# impute missing year with "median"
median_imputer = SimpleImputer(strategy="median")
df["Year"] = median_imputer.fit_transform(df[["Year"]])

# impute missing publisher with "mode"
mode_imputer = SimpleImputer(strategy="most_frequent")
df["Publisher"] = mode_imputer.fit_transform(df[["Publisher"]]).ravel()

# Feature Engineering - Create "Game_Age"

current_year = 2026
df["Game_Age"] = current_year - df['Year']

numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
categorical_features = df.select_dtypes(include=["object"]).columns


# Outlier Detection

for feature in numeric_features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
    df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])


# Encoding Categorical Features

# Initialize OneHotEncoder
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Apply OneHotEncoder to the categorical features
encoded_features = one_hot_encoder.fit_transform(df[categorical_features])

# Create a DataFrame from the encoded features
encoded_feature_names = one_hot_encoder.get_feature_names_out(
    categorical_features)
df_encoded = pd.DataFrame(
    encoded_features, columns=encoded_feature_names, index=df.index)

# Drop original categorical columns and concatenate the encoded features
df = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)


# Scaling Numerical Features
numerical_features_to_scale = [
    col for col in numeric_features if col != target_col]

scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(
    df[numerical_features_to_scale])

# Train test split
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#  Pipeline Creation


current_numerical_features = df.select_dtypes(
    include=["int64", "float64"]).columns
current_categorical_features = df.select_dtypes(include=["object"]).columns

# Exclude target_col from numerical features for the pipeline
numerical_features_for_pipeline = [
    col for col in current_numerical_features if col != target_col]
categorical_features_for_pipeline = list(current_categorical_features)

# pipeline for numerical transformations
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create a pipeline for categorical transformations
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_for_pipeline),
        ('cat', categorical_transformer, categorical_features_for_pipeline)
    ],
    remainder='passthrough'
)


# final model
final_best_model = RandomForestRegressor(
    n_estimators=57,
    max_depth=20,
    min_samples_split=8,
    random_state=42
)

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', final_best_model)
])


rf_model.fit(X_train, y_train)


y_pred_final = rf_model.predict(X_test)


r2_final = r2_score(y_test, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
mae_final = mean_absolute_error(y_test, y_pred_final)

print(f"Final Model Performance on Test Set:")
print(f"R-squared: {r2_final:.4f}")
print(f"RMSE: {rmse_final:.4f}")
print(f"MAE: {mae_final:.4f}")


# Save model

with open("video_games_sales.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("✅ Random Forest pipeline saved as video_games_sales.pkl")
