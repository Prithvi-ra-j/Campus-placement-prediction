from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model (ensure this file exists in the same directory or provide the correct path)
model = joblib.load('pipeline.pkl')

# Get the model's expected feature columns (for debugging)
model_columns = model.feature_names_in_

@app.route('/')
def home():
    """
    Home page that renders the form.
    """
    return render_template('form.html')  # Serve the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission and make predictions.
    """
    try:
        # Extract form data
        ssc_p = request.form.get('ssc_p')
        hsc_p = request.form.get('hsc_p')
        degree_p = request.form.get('degree_p')
        etest_p = request.form.get('etest_p')
        hsc_s = request.form.get('hsc_s')
        degree_t = request.form.get('degree_t')

        # Check for missing or invalid values in the form
        if not all([ssc_p, hsc_p, degree_p, etest_p, hsc_s, degree_t]):
            return "Error: Missing input fields. Please provide all required values.", 400
        
        # Convert form data to float, with error handling for invalid numerical input
        try:
            ssc_p = float(ssc_p)
            hsc_p = float(hsc_p)
            degree_p = float(degree_p)
            etest_p = float(etest_p)
        except ValueError:
            return "Error: Invalid numerical input. Please ensure all numerical fields contain valid numbers.", 400

        # One-hot encode categorical variables
        hsc_s_arts = 1 if hsc_s == 'Arts' else 0
        hsc_s_commerce = 1 if hsc_s == 'Commerce' else 0
        hsc_s_science = 1 if hsc_s == 'Science' else 0
        
        degree_t_comm_mgmt = 1 if degree_t == 'Comm&Mgmt' else 0
        degree_t_others = 1 if degree_t == 'Others' else 0
        degree_t_sci_tech = 1 if degree_t == 'Sci&Tech' else 0

        # Construct the feature vector
        feature_data = [
            ssc_p, hsc_p, degree_p, etest_p,
            hsc_s_arts, hsc_s_commerce, hsc_s_science,
            degree_t_comm_mgmt, degree_t_others, degree_t_sci_tech
        ]

        # Log the input data to the console
        print("Received Input Data:")
        print(f"ssc_p: {ssc_p}, hsc_p: {hsc_p}, degree_p: {degree_p}, etest_p: {etest_p}")
        print(f"hsc_s: {hsc_s}, degree_t: {degree_t}")
        print(f"One-Hot Encoded Features: hsc_s_Arts: {hsc_s_arts}, hsc_s_Commerce: {hsc_s_commerce}, hsc_s_Science: {hsc_s_science}")
        print(f"degree_t_Comm&Mgmt: {degree_t_comm_mgmt}, degree_t_Others: {degree_t_others}, degree_t_Sci&Tech: {degree_t_sci_tech}")
        print(f"Feature Array: {feature_data}")

        # Ensure the input data matches the model's feature columns
        input_data = dict(zip(model_columns, feature_data))

        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure the input matches the columns expected by the model
        missing_cols = set(model_columns) - set(input_df.columns)
        if missing_cols:
            return f"Error: Missing columns in the input data: {missing_cols}", 400

        # Log the DataFrame to the console
        print(f"Input DataFrame:\n{input_df}")

        # Make the prediction using the trained model
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df).tolist()

        # Log the prediction result
        print(f"Prediction: {prediction}")
        print(f"Prediction Probability: {prediction_prob}")

        # If the model outputs strings (e.g., 'Placed', 'Not Placed'), convert them to integers
        if isinstance(prediction, str):
            if prediction == 'Placed':
                prediction = 1
            elif prediction == 'Not Placed':
                prediction = 0
            else:
                raise ValueError("Unexpected prediction output.")

        # Return the results page with the prediction and probabilities
        return render_template('result.html', 
                               prediction=int(prediction), 
                               probability=prediction_prob)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
