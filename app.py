from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('heart_disease_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json(force=True)
        
        # Convert input data to a numpy array
        input_data_np = np.asarray(input_data)
        
        # Reshape the numpy array
        input_data_reshaped = input_data_np.reshape(1, -1)
        
        # Make predictions
        prediction = model.predict(input_data_reshaped)
        
        # Return the prediction as JSON
        result = {'prediction': int(prediction[0])}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

