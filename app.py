from flask import Flask, request, jsonify
import torch
import pickle
import pandas as pd

# Load the saved model, tokenizer, and label encoder
with open('BERT.pkl', 'rb') as model_file:
    tokenizer, data, model, label_encoder = pickle.load(model_file)

# Set the model to evaluation mode
model.eval()

# Predict function
def predict(text, tokenizer, model, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=1).item()
    return label_encoder.inverse_transform([predicted_label])[0], predictions.tolist()

# Initialize Flask app
app = Flask(__name__)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Parse incoming JSON data
        data = request.get_json()

        # Validate the input
        if 'text' not in data:
            return jsonify({"error": "Missing 'text' field in the request"}), 400
        
        user_input = data['text']
        if not user_input.strip():
            return jsonify({"error": "Input text cannot be empty"}), 400

        # Perform prediction
        predicted_label, predictions = predict(user_input, tokenizer, model, label_encoder)

        # Format predictions as a DataFrame-like dictionary
        prediction_table = pd.DataFrame([predictions], columns=["Negative", "Neutral", "Positive"]).to_dict(orient="records")[0]

        # Prepare response
        response = {
            "input_text": user_input,
            "predicted_label": predicted_label,
            "prediction_probabilities": prediction_table
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
