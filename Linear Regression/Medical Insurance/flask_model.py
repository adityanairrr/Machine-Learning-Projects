from flask import Flask, render_template, request
from model_logic import get_prediction  # Importing from your separate file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capture form data in your specific order
        # Age, Gender, BMI, Children, Smoker, Region
        user_data = [
            float(request.form['age']),
            float(request.form['gender']),
            float(request.form['bmi']),
            float(request.form['children']),
            float(request.form['smoker']),
            float(request.form['region'])
        ]
        
        # Call the logic from the other file
        result = get_prediction(user_data)
        
        return render_template('index.html', 
                               prediction_text=f'Estimated Insurance Cost: Rs{result:,.2f}')
    
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f"Error: Please check your inputs. ({str(e)})")

if __name__ == "__main__":
    app.run(debug=True)