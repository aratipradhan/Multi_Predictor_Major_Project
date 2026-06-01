import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and clean data
df = pd.read_csv(r"D:\Arati_Project\Multi-Predictor App\Data Set\cleaned_car_data.csv")
df.dropna(inplace=True)

# Select features and target
features = ['Year', 'Engine HP', 'Engine Cylinders', 'Transmission Type', 'Driven_Wheels',
            'city mpg', 'highway MPG', 'Popularity']
X = df[features]
y = df['Selling_Price']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=['Transmission Type', 'Driven_Wheels'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and columns
with open(r"D:\Arati_Project\Multi-Predictor App\models\car_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(r"D:\Arati_Project\Multi-Predictor App\models\model_columns.pkl", "wb") as f:
    pickle.dump(X_encoded.columns.tolist(), f)

print("✅ Car model and columns saved.")
