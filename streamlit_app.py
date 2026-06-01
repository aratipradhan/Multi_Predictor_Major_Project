import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

# -------------------------------
# Load Lottie animation from URL
# -------------------------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

# -------------------------------
# Load models
# -------------------------------
house_model = pickle.load(open(r"D:\Arati_Project\Multi-Predictor App\models\house_model.pkl", 'rb'))


salary_model = pickle.load(open(r"D:\Arati_Project\Multi-Predictor App\models\salary.pkl", 'rb'))
investment_model = pickle.load(open(r"D:\Arati_Project\Multi-Predictor App\models\Investment_Model.pkl", 'rb'))
investment_scaler = pickle.load(open(r"D:\Arati_Project\Multi-Predictor App\models\Investment_Scaler.pkl", 'rb'))
car_model = pickle.load(open(r"D:\Arati_Project\Multi-Predictor App\models\car_model.pkl", 'rb'))
model_columns = pickle.load(open(r"D:\Arati_Project\Multi-Predictor App\models\model_columns.pkl", 'rb'))

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Multi-Predictor App", layout="centered")

# -------------------------------
# Custom Sidebar Styling
# -------------------------------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .sidebar-title {
        font-size: 28px;
        font-weight: bold;
        color: #4B8BBE;
        margin-bottom: 10px;
    }
    .sidebar-subtitle {
        font-size: 16px;
        color: #333333;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Layout with Option Menu
# -------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">📊 Multi-Predictor App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Your one-stop solution for predictive insights.</div>', unsafe_allow_html=True)

    sidebar_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
    if sidebar_lottie:
        st_lottie(sidebar_lottie, height=150)

    st.markdown("---")

    option = option_menu(
    menu_title="Prediction Models",
    options=["🏠 House Price", "💰 Investment", "💼 Salary", "🚗 Car Price", "ℹ️ About Us"],
    icons=["house", "graph-up", "briefcase", "car-front", "info-circle"],
    menu_icon="cast",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "#4B8BBE", "font-size": "20px"},
        "nav-link": {"font-size": "18px", "text-align": "left", "margin": "2px", "--hover-color": "#e0e0e0"},
        "nav-link-selected": {"background-color": "#4B8BBE", "color": "white"},
    }
)


# -------------------------------
# Load Section Animation
# -------------------------------
if option == "🏠 House Price":
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_cwqf3pdx.json"
elif option == "💰 Investment":
    lottie_url = "https://assets7.lottiefiles.com/packages/lf20_qp1q7mct.json"
elif option == "💼 Salary":
    lottie_url = "https://assets3.lottiefiles.com/packages/lf20_hdy0htc2.json"
else:
    lottie_url = "https://assets10.lottiefiles.com/packages/lf20_yfhvla3d.json"

animation = load_lottieurl(lottie_url)
if animation:
    st_lottie(animation, height=200)

# -------------------------------
# House Price Prediction
# -------------------------------
if option == "🏠 House Price":
    st.header("🏠 House Price Prediction")
    sqft = st.number_input("Square Feet", 400, 10000, step=100)
    floor = st.slider("Floors", 0, 10)
    bedroom = st.slider("Bedrooms", 0, 30)
    bathroom = st.slider("Bathrooms", 0, 10)
    condition = st.slider("Condition (0 = Poor, 5 = Excellent)", 0, 5)

    if st.button("Predict House Price"):
        input_data = np.array([[sqft, floor, bedroom, bathroom, condition]])
        prediction = house_model.predict(input_data)
        st.success(f"Estimated House Price: ₹{prediction[0]:,.2f}")

# -------------------------------
# Investment ROI Prediction
# -------------------------------
elif option == "💰 Investment":
    st.header("💰 Investment ROI Prediction")
    st.write("Predict your Profit from Digital Marketing, Promotions, and Research investments.")

    digital_marketing = st.number_input("Digital Marketing (₹)", 0, 1000000, step=100)
    promotion = st.number_input("Promotion (₹)", 0, 200000, step=100)
    research = st.number_input("Research (₹)", 0, 1000000, step=100)

    if st.button("Predict ROI"):
        input_data = np.array([[digital_marketing, promotion, research]])
        scaled_input = investment_scaler.transform(input_data)
        prediction = investment_model.predict(scaled_input)
        st.success(f"Predicted Profit: ₹{prediction[0]:,.2f}")
        st.write("This is an estimate based on your inputs.")

# -------------------------------
# Salary Prediction
# -------------------------------
elif option == "💼 Salary":
    st.header("💼 Salary Prediction")
    experience = st.number_input("Years of Experience", 0.0, 50.0, 1.0, step=0.5)

    if st.button("Predict Salary"):
        prediction = salary_model.predict(np.array([[experience]]))
        st.success(f"Estimated Salary: ₹{prediction[0]:,.2f}")

# -------------------------------
# Car Price Prediction
# -------------------------------
elif option == "🚗 Car Price":
    st.header("🚗 Car Price Prediction")
    year = st.slider("Year", 1990, 2025, 2015)
    engine_hp = st.number_input("Engine HP", 50, 1000, 150)
    engine_cylinders = st.selectbox("Engine Cylinders", [3, 4, 6, 8])
    transmission = st.selectbox("Transmission", ["MANUAL", "AUTOMATIC"])
    drive = st.selectbox("Driven Wheels", ["front wheel drive", "rear wheel drive", "all wheel drive", "four wheel drive"])
    city_mpg = st.number_input("City MPG", 5, 50, 15)
    highway_mpg = st.number_input("Highway MPG", 10, 60, 20)
    popularity = st.slider("Popularity", 100, 5000, 1000)

    input_dict = {
        'Year': year,
        'Engine HP': engine_hp,
        'Engine Cylinders': engine_cylinders,
        'city mpg': city_mpg,
        'highway MPG': highway_mpg,
        'Popularity': popularity,
        f'Transmission Type_{transmission}': 1,
        f'Driven_Wheels_{drive}': 1
    }

    df_input = pd.DataFrame([input_dict])
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    st.write("📊 Input Preview:")
    st.dataframe(df_input)

    if st.button("Predict Car Price"):
        prediction = car_model.predict(df_input)
        price = max(prediction[0], 0)
        st.success(f"Estimated Selling Price: ₹{price:,.2f}")

# -------------------------------
# About Us Section
# -------------------------------
elif option == "ℹ️ About Us":
    st.header("ℹ️ About Us")
   

    st.markdown("""
    ### Welcome to **Multi-Predictor App**!

    We are a passionate team building intelligent tools for real-world predictions:
    
    - 🏠 **House Price Estimator**
    - 💰 **Investment ROI Forecaster**
    - 💼 **Salary Predictor**
    - 🚗 **Car Price Evaluator**

    Our goal is to simplify decision-making using machine learning.
    
    ### 👩‍💻 Developed By:
    - K. Arati Pradhan (REg.- 2101220028, Dept- CSE)
    - Sanket Maharana (REg.- 2101220044, Dept- CSE)
    - Mahesh KrishnaMurti (REg.- 2221220014, Dept- CSE)
    - Sahil Badtya (REg.- 2101220042, Dept- CSE)
    

     ### 🔧 Tools Used:
    - Python
    - Streamlit
    - scikit-learn
    - Machine Learning Models

    ### 📌 Purpose:
    Our goal is to make decisions **easier, faster, and accessible** to everyone.  
   
🔔 **Note:** This is for educational purposes only and not a substitute for professional financial advice.
</div>
""", unsafe_allow_html=True)


st.markdown("""
<hr style="border: none; border-top: 1px solid #eee;" />
<div style='text-align: center; font-size: 14px; color: gray;'>
© 2025 <a href='https://www.Multi-Predictor App.com' target='_blank'>Multi-Predictor App</a>. All rights reserved.
</div>
""", unsafe_allow_html=True)
