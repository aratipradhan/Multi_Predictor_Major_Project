import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example dataset (you should replace this with your actual house dataset)
# Make sure it includes: Square Feet, Floors, Bedrooms, Bathrooms, Condition, and Price
df = pd.read_csv(r"D:\Arati_Project\Multi-Predictor App\Data Set\house_data.csv")

#df.columns = df.columns.str.strip()  # 🧼 Remove any leading/trailing spaces

# 🛠 Print column names to debug if needed
print("Columns in dataset:", df.columns)

# ✅ Adjust these feature names to match your CSV exactly
X = df[['sqft_living', 'floors', 'bedrooms', 'bathrooms', 'condition', 'price']]
y = df['price']

# 🔀 Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 💾 Save the model to a relative folder for Streamlit Cloud
import os
os.makedirs("models", exist_ok=True)

with open("models/house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ House model trained and saved to models/house_model.pkl")