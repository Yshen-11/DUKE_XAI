# ===============================
# Import necessary libraries
# ===============================
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ===============================
# 1. Data Loading and Preprocessing
# ===============================
def load_and_preprocess_data(file_path):
    """
    Load data and preprocess it, including handling missing values and feature engineering.
    Args:
        file_path (str): Path to the data file
    Returns:
        pd.DataFrame: Processed data
        LabelEncoder: Encoder object
    """
    # Load data
    data = pd.read_csv(file_path)

    # Select key features
    selected_features = ['Pk', 'College', 'MPG', 'PPG', 'RPG', 'APG']
    data = data[selected_features].copy()  # Use .copy() to create a copy

    # Handle missing values in numeric and categorical columns
    numeric_columns = ['MPG', 'PPG', 'RPG', 'APG']
    
    # Handle missing values in the College column
    data['College'].fillna("Unknown", inplace=True)
    
    # Handle missing values in numeric columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    # Encode textual features
    encoder = LabelEncoder()
    data['College'] = encoder.fit_transform(data['College'])

    # Save the encoder
    joblib.dump(encoder, "college_encoder.pkl")
    print("LabelEncoder saved as college_encoder.pkl")

    return data

# ===============================
# 2. Data Splitting and Standardization
# ===============================
def prepare_data(data):
    """
    Split features and target variable and standardize the data.
    Args:
        data (pd.DataFrame): Processed data
    Returns:
        X_train, X_test, y_train, y_test, scaler: Split data and scaler
    """
    X = data.drop(columns=['Pk'])  # Features
    y = data['Pk']  # Target variable

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler

# ===============================
# 3. Random Forest Model Tuning
# ===============================
def train_and_tune_model(X_train, y_train):
    """
    Tune the Random Forest model using GridSearchCV.
    Args:
        X_train: Training features
        y_train: Training target
    Returns:
        best_model: Tuned best Random Forest model
    """
    # Define parameter grid
    param_grid_rf = {
        'n_estimators': [100, 200, 300],       # Number of decision trees
        'max_depth': [10, 15, 20],            # Maximum depth
        'min_samples_split': [2, 5, 10],      # Minimum samples required to split a node
        'max_features': ['auto', 'sqrt']      # Maximum number of features per tree
    }

    # Use GridSearchCV for tuning
    grid_search_rf = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid_rf,
        scoring='neg_mean_squared_error',     # Use negative MSE as scoring metric
        cv=3,                                # 3-fold cross-validation
        verbose=1
    )

    # Fit the model
    grid_search_rf.fit(X_train, y_train)

    # Output the best parameters and model
    print(f"Best Random Forest Parameters: {grid_search_rf.best_params_}")
    return grid_search_rf.best_estimator_

# ===============================
# 4. Model Evaluation and Saving
# ===============================
def evaluate_and_save_model(model, X_test, y_test, scaler):
    """
    Evaluate the model and save the results.
    Args:
        model: Tuned model
        X_test: Test features
        y_test: Test target
        scaler: Data scaler
    """
    # Make predictions with the best model
    predictions = model.predict(X_test)

    # Calculate model performance
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Optimized Random Forest MSE: {mse:.2f}, R²: {r2:.2f}")

    # Save the model and scaler
    joblib.dump(model, "best_rf_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved!")

# ===============================
# Main Program Entry
# ===============================
if __name__ == "__main__":
    # Load and preprocess data
    file_path = 'draft-data-20-years.csv'  # Replace with actual path
    data = load_and_preprocess_data(file_path)

    # Split and standardize data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)

    # Train and tune the model
    best_rf_model = train_and_tune_model(X_train, y_train)

    # Evaluate and save the model
    evaluate_and_save_model(best_rf_model, X_test, y_test, scaler)
