import streamlit as st
import pickle
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Obesity Risk Analyzer", layout="wide")

# --- CUSTOM CSS (To make it look like your HTML site) ---
st.markdown("""
    <style>
    /* Background Gradient */
    .stApp {
        background: linear-gradient(to right, #e0eafc, #cfdef3);
    }
    /* Prediction Card Styling */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 10px solid #ff4b4b;
    }
    .result-text {
        font-size: 30px;
        font-weight: bold;
        color: #1e3d59;
    }
    .github-link {
        position: fixed;
        bottom: 20px;
        right: 20px;
        text-decoration: none;
        transition: transform 0.3s ease;
        z-index: 1000;
    }
    .github-link:hover {
        transform: scale(1.2);
    }
    .github-icon {
        width: 40px;
        height: 40px;
        filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.2));
    }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "Obesity_check_model.pkl")

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    return None

model = load_model()

# --- HEADER ---
st.write("# ⚖️ Obesity Level Prediction")
st.write("---")

if model is None:
    st.error("Error: 'model.pkl' not found in the directory!")
else:
    # --- FORM ---
    with st.form("prediction_form"):
        st.subheader("Personal & Lifestyle Metrics")
        
        # Row 1: Basics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            Gender = st.selectbox("Gender", ["Female", "Male"])
            Age = st.number_input("Age", 1, 100, 25)
        with col2:
            Height = st.number_input("Height (m)", 1.2, 2.2, 1.7)
            Weight = st.number_input("Weight (kg)", 30, 200, 70)
        with col3:
            Familyoverweight_history = st.selectbox("Family History?", ["yes", "no"])
            HighCaloriefood_consumption = st.selectbox("High Calorie Food Consumption?", ["yes", "no"])
        with col4:
            Consumption_of_vegetables = st.slider("Veggie Intake (1-3)", 1.0, 3.0, 2.0)
            No_of_mainmeals = st.slider("Main Meals / Day", 1.0, 4.0, 3.0)

        st.write("---")
        st.subheader("Eating & Activity Habits")
        
        # Row 2: Habits
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            food_between_meals = st.selectbox("Eat between meals?", ["no", "Sometimes", "Frequently", "Always"])
            Smoking = st.selectbox("Do you smoke?", ["no", "yes"])
        with col6:
            CH20 = st.slider("Water Intake (Liters)", 1.0, 3.0, 2.0)
            Beverages_consumption= st.selectbox("Monitor Calories?", ["no", "yes"])
        with col7:
            Physical_Activity = st.slider("Physical Activity (0-3)", 0.0, 3.0, 1.0)
            Time_on_devices = st.slider("Tech Usage (0-2 hours)", 0.0, 2.0, 1.0)
        with col8:
            Alcohol = st.selectbox("Alcohol Intake", ["no", "Sometimes", "yes"])
            Mode_of_transportation = st.selectbox("Transport Mode", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

        # --- PREDICTION LOGIC ---
        submit = st.form_submit_button("ANALYZE RISK LEVEL")

        if submit:
            # Mapping inputs to Match Model Training
            Gender_val = 1 if Gender == "Male" else 0
            Familyoverweight_history_val = 1 if Familyoverweight_history == "yes" else 0
            HighCaloriefood_consumption_val = 1 if  HighCaloriefood_consumption == "yes" else 0
            Smoking_val = 1 if Smoking == "yes" else 0
            Beverages_consumption_val = 1 if Beverages_consumption == "yes" else 0
            
            # Categorical Mapping for CAEC, CALC, MTRANS
            food_between_meals_map = {"no": 0, "Sometimes": 3, "Frequently": 2, "Always": 1}
            Alcohol_map = {"Frequently": 1, "no": 0, "Sometimes": 2}
            Mode_of_transportation_map = {"Automobile": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Walking": 4}

            # ARRANGE ALL 16 FEATURES (Order must match your model training!)
            features = np.array([[
                Gender_val, Age,Height, Weight,  Familyoverweight_history_val, HighCaloriefood_consumption_val, 
                Consumption_of_vegetables,No_of_mainmeals, food_between_meals_map[ food_between_meals], Smoking_val, CH20, Beverages_consumption_val, 
                Physical_Activity, Time_on_devices,  Alcohol_map[Alcohol], Mode_of_transportation_map[Mode_of_transportation]
            ]])

# --- 1. GET THE PREDICTION ---
            raw_prediction = model.predict(features)
            res = raw_prediction[0]  # Extract the string (e.g., 'Normal_Weight')

            # --- 2. SET THE COLOR ---
            if "Obesity" in res:
                bg_color = "#e74c3c"  # Red
            elif "Overweight" in res:
                bg_color = "#f39c12"  # Orange
            elif "Normal" in res:
                bg_color = "#2ecc71"  # Green
            elif "Insufficient" in res or "Underweight" in res:
                bg_color = "#3498db"  # Blue
            else:
                bg_color = "#95a5a6"  # Gray

            st.write("---")
            
            # --- 3. DISPLAY THE COLOR BOX ---
            # We use .replace('_', ' ') so 'Normal_Weight' becomes 'Normal Weight'
            st.markdown(f"""
                <div style="
                    background-color: {bg_color}; 
                    padding: 30px; 
                    border-radius: 15px; 
                    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
                    text-align: center;">
                    <h3 style="color: white; margin-bottom: 5px; font-family: sans-serif; font-weight: 400;">Predicted Health Category</h3>
                    <h1 style="color: white; margin-top: 0; font-size: 45px; font-family: sans-serif;">{res.replace('_', ' ')}</h1>
                </div>
            """, unsafe_allow_html=True)

            st.balloons()
            
            # You can keep this simple text prediction if you want it as a backup
            st.info(f"Analysis complete for user.")

# --- 4. GITHUB ICON (Keep this OUTSIDE the 'if submit' block so it's always visible) ---
st.markdown(f"""
    <a href="https://github.com/adityanairrr/Machine-Learning-Projects" target="_blank" class="github-link">
        <img src="https://cdns.iconmonstr.com/wp-content/releases/preview/2012/240/iconmonstr-github-1.png" class="github-icon">
    </a>
""", unsafe_allow_html=True)
