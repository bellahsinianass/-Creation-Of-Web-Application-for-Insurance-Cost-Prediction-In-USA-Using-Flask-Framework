from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__,template_folder="template")

# Load the saved model using pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Create a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    sex = request.form['sex']
    smoker = request.form['smoker']
    region = request.form['region']

    # Convert the categorical values to binary values
    sex_female = 1 if sex == 'female' else 0
    sex_male = 1 if sex == 'male' else 0
    smoker_no = 1 if smoker == 'no' else 0
    smoker_yes = 1 if smoker == 'yes' else 0

    # Convert the region to binary values
    region_map = {
        'northeast': [1, 0, 0, 0],
        'northwest': [0, 1, 0, 0],
        'southeast': [0, 0, 1, 0],
        'southwest': [0, 0, 0, 1]
    }
    region_values = region_map.get(region, [0, 0, 0, 0])

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex_female': [sex_female],
        'sex_male': [sex_male],
        'smoker_no': [smoker_no],
        'smoker_yes': [smoker_yes],
        'region_northeast': [region_values[0]],
        'region_northwest': [region_values[1]],
        'region_southeast': [region_values[2]],
        'region_southwest': [region_values[3]]
    })

    # Make the prediction
    prediction = model.predict(input_data)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
