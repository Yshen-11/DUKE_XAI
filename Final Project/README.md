Here is the complete `README.md` file entirely in Markdown format:

```markdown
# **NBA Draft Prediction Project**

## **Project Overview**
This is a Flask-based web application designed to predict the NBA draft position of players based on their college performance and background. The project uses a `RandomForestRegressor` model and incorporates `SHAP` to provide feature importance explanations, helping users understand the predictions.

## **Directory Structure**
```plaintext
FINAL PROJECT/
├── templates/                # Front-end HTML templates
│   └── index.html
├── app.py                    # Main Flask application
├── best_rf_model.pkl         # Trained Random Forest model
├── college_encoder.pkl       # LabelEncoder for encoding the "College" feature
├── draft-data-20-years.csv   # Original dataset
├── model.py                  # Data preprocessing and model training script
├── scaler.pkl                # StandardScaler object for data normalization
├── README.md                 # Project documentation
```

## **Environment Setup**
- **Python Version**: 3.11
- **Environment Manager**: Conda

## **Installation Steps**

### 1. **Create a Virtual Environment**
Run the following commands to create and activate a virtual environment:
```bash
conda create -n nba_env python=3.11
conda activate nba_env
```

### 2. **Install Required Libraries**
Install the required packages in the virtual environment:
```bash
conda install scikit-learn=1.5.2 flask pandas shap joblib
```

### 3. **Clone or Download the Project**
Ensure the project directory includes the following files and folders:
- `app.py`
- `model.py`
- Pre-trained files (`best_rf_model.pkl`, `scaler.pkl`, `college_encoder.pkl`)
- Dataset (`draft-data-20-years.csv`)

### 4. **Train the Model (Optional)**
To retrain the model, run `model.py`:
```bash
python model.py
```
Upon completion, the following files will be generated:
- `best_rf_model.pkl`
- `scaler.pkl`
- `college_encoder.pkl`

### 5. **Run the Application**
Start the Flask application:
```bash
python app.py
```

### 6. **Access the Application**
Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000). Enter player information to get predictions and feature importance explanations.

## **Features**
1. **User Input**:
   - Users can input player information, including college and per-game statistics (e.g., `MPG`, `PPG`, `RPG`, `APG`).
2. **Prediction Results**:
   - The application predicts the NBA draft position based on the input data.
3. **Feature Explanation**:
   - `SHAP` visualizations explain the contribution of each feature to the prediction.

## **Tech Stack**
- **Backend**: Flask
- **Machine Learning**: Scikit-learn
- **Interpretability**: SHAP
- **Frontend**: HTML + JavaScript (Chart.js)

## **Limitations**
- **Dataset Issues**:
  - The current dataset lacks information about players' physical attributes and detailed college game statistics.
  - A more comprehensive dataset, including players' physical measurements (e.g., height, weight, wingspan) and advanced college game data, would greatly enhance prediction accuracy.

## **Future Improvements**
- Integrate a better dataset with physical and performance statistics.
- Explore additional models such as XGBoost or LightGBM for improved performance.
- Add real-time data scraping for the latest player information.

