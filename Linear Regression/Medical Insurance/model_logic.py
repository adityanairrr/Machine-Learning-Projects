import pickle
import numpy as np

def get_prediction(data_list):
    # Load your model (ensure the pkl file is in the same folder)
    with open('medical_insurance_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Convert input to 2D array
    input_array = np.array([data_list])
    
    # Predict
    prediction = model.predict(input_array)
    
    # Return formatted result
    return round(float(prediction[0]), 2)