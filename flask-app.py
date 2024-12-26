from datetime import datetime

from flask import Flask, request, jsonify, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Directory to save images
UPLOAD_FOLDER = './saved-photos'
STATIC_UPLOAD_FOLDER = "./static/images"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_UPLOAD_FOLDER, exist_ok=True)

# Load the trained ML model
MODEL_PATH = 'plant_disease_model.h5'
model = load_model(MODEL_PATH)

# Adjust according to your dataset (Make sure this matches your model's output)
CLASS_NAMES = ['Healthy_Bean', 'Diseased', 'Healthy_Bean']  # Update with correct class names

# In-memory persistent storage for predictions

predictions = {
    "production": "No Prediction Yet",
    "confidence": "No Confidence Yet"
}

@app.get("/")
def home():
    return "HOME"

@app.route('/image')
def get_image():
    # Directory where the image is stored
    image_dir = STATIC_UPLOAD_FOLDER  # Ensure this points to the 'static' directory

    try:
        # Check if the directory exists and is not empty
        if not os.path.exists(image_dir) or not os.listdir(image_dir):
            return jsonify({"error": "No images found"}), 404

        # Get the latest image in the directory
        images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        if not images:
            return jsonify({"error": "No images found"}), 404

        # Sort by creation time and select the most recent image
        latest_image = max(images, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))

        # Build the URL to the static image
        image_url = url_for('static', filename=f'images/{latest_image}', _external=True)

        # Return the image URL and prediction details
        return jsonify({
            "image": image_url,
            "pred": predictions.get("production", "No prediction available"),
            "conf": predictions.get("confidence", "No confidence available")
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route('/saved-photo', methods=['POST'])
def saved_photo():
    try:
        # Ensure the request contains image data
        if request.content_type != 'image/jpeg':
            return "Invalid Content-Type. Expecting 'image/jpeg'", 400

        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"image_{timestamp}.jpg"

        # Save the uploaded image
        image_data = request.data
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        static_img_path = os.path.join(STATIC_UPLOAD_FOLDER, image_filename)

        # Delete existing images in the folder
        for file in os.listdir(UPLOAD_FOLDER):
            if file.endswith('.jpg'):
                os.remove(os.path.join(UPLOAD_FOLDER, file))

        for file in os.listdir(STATIC_UPLOAD_FOLDER):
            if file.endswith('.jpg'):
                os.remove(os.path.join(STATIC_UPLOAD_FOLDER, file))

        with open(image_path, 'wb') as f:
            f.write(image_data)

        with open(static_img_path, 'wb') as f:
            f.write(image_data)

        print(f"Image saved at {image_path}")
        img_path = image_path
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Unable to load image from path: {img_path}")

        # Resize image to the required model input size (256, 256 in your case)
        img_resized = cv2.resize(img, (256, 256))

        # Convert to an array and normalize
        img_array = image.img_to_array(img_resized) / 255.0  # Normalize to [0, 1]

        # Add batch dimension for model input
        img_array = np.expand_dims(img_array, axis=0)

        print("Model input shape:", model.input_shape)

        # Make prediction with the model
        prediction = model.predict(img_array)

        # Check the shape and content of prediction to debug
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction: {prediction}")

        # Assuming a classification model, process the prediction
        if prediction.ndim > 1 and prediction.shape[1] > 0:
            predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the scalar value from the array
            if predicted_class_index < len(CLASS_NAMES):
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100
                print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}%")
            else:
                raise IndexError(f"Predicted class index {predicted_class_index} is out of range.")
        else:
            raise ValueError("Prediction output is malformed.")

        # Update persistent storage with the new prediction
        predictions["production"] = predicted_class
        predictions["confidence"] = f"{confidence:.2f}%"

        response = {
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%"
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
