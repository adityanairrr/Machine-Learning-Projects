import streamlit as st
import pickle
import numpy as np

# 1. Load the Model
@st.cache_resource
def load_model():
    # Make sure 'model.pkl' is in the same folder as this app.py
    with open('Obesity_check_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# 2. Define Mappings (Match these to your specific model training)
mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'Familyoverweight_history': {'No': 0, 'Yes': 1},
    'HighCaloriefood_consumption': {'No': 0, 'Yes': 1},
    'food_between_meals': {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
    'Smoking': {'No': 0, 'Yes': 1},
    'Beverages_consumption': {'No': 0, 'Yes': 1},
    'Alcohol': {'No': 0, 'Sometimes': 1, 'Frequently': 2},
    'Mode_of_transportation': {
        'Automobile': 0, 'Bike': 1, 'Motorbike': 2, 
        'Public_Transportation': 3, 'Walking': 4
    }
}

# 3. UI Styling
st.title("⚖️ HealthSense ")
st.write("Obesity Risk Analysis based on Lifestyle Metrics")

# 4. Input Form
with st.form("main_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", 1, 100, 25)
        height = st.number_input("Height (m)", 1.0, 2.5, 1.75)
        weight = st.number_input("Weight (kg)", 10.0, 250.0, 70.0)
    with col2:
        family = st.selectbox("Family Overweight History?", ["Yes", "No"])
        calc_intake = st.selectbox("High Calorie Food?", ["Yes", "No"])
        veg = st.slider("Veggie Consumption (1-3)", 1.0, 3.0, 2.0)
        meals = st.slider("Main Meals Per Day", 1.0, 4.0, 3.0)

    # Add other inputs here if your model requires all 16 features...
    
    submit = st.form_submit_button("Predict Obesity Level")

# 5. Prediction
if submit:
    # This list must match the order of features your model was trained on!
    features = np.array([[
        mappings['Gender'][gender], age, height, weight,
        mappings['Familyoverweight_history'][family],
        mappings['HighCaloriefood_consumption'][calc_intake],
        veg, meals, 1, 0, 2.0, 0, 1.0, 1.0, 1, 3 # Example placeholders for remaining features
    ]])
    
    prediction = model.predict(features)[0]
    st.success(f"Result: {prediction}")

    except Exception as e:
        return f"Error: {str(e)}. Please check if all fields are filled correctly."

if __name__ == "__main__":
    app.run(debug=True)
