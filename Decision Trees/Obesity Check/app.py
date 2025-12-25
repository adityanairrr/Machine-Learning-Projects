from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 1. Load your model
# Ensure your model file is in the same folder as this script
try:
    with open('Obesity_check_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: model.pkl not found.")

# 2. Input Mappings (Converts HTML text to numbers for the model)
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        f = request.form
        
        # Construct the feature list in the EXACT order your model expects
        # Note: We convert text to numbers using the mappings above
        features = [
            mappings['Gender'][f['Gender']],
            float(f['Age']),
            float(f['Height']),
            float(f['Weight']),
            mappings['Familyoverweight_history'][f['Familyoverweight_history']],
            mappings['HighCaloriefood_consumption'][f['HighCaloriefood_consumption']],
            float(f['Consumption_of_vegetables']),
            float(f['No_of_mainmeals']),
            mappings['food_between_meals'][f['food_between_meals']],
            mappings['Smoking'][f['Smoking']],
            float(f['WaterConsumption']),
            mappings['Beverages_consumption'][f['Beverages_consumption']],
            float(f['Physical_Activit']),
            float(f['Time_on_devices']),
            mappings['Alcohol'][f['Alcohol']],
            mappings['Mode_of_transportation'][f['Mode_of_transportation']]
        ]

        # Convert to numpy array and predict
        final_features = np.array([features])
        prediction = model.predict(final_features)

        # Since your model returns a category directly:
        result = prediction[0] 

        return render_template('index.html', prediction_text=f'Predicted Obesity Level: {result}')

    except Exception as e:
        return f"Error: {str(e)}. Please check if all fields are filled correctly."

if __name__ == "__main__":
    app.run(debug=True)