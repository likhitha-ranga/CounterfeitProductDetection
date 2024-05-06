import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory
import uuid
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model_path = 'fakeProduct.h5'  # Update with your model's path
model = load_model(model_path,compile=False)

# Set up the image upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a function to make predictions
def preprocess_image(img_path):
    target_size = (224, 224)  # Ensure this matches the model's expected input shape
    img = image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    class_labels = ['Fake Product', 'Real Product']
    # Predict and get class name and confidence
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)
    if predicted_class==0:
        msg="It is not advisable to buy the product..!"
    else:
        msg="You can buy the product for top-notch quality and value!"
    return class_labels[predicted_class], confidence , msg


# Route for the home page to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # This file will contain your HTML form

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Save the uploaded image to the upload folder
    file = request.files['file']
    filename = str(uuid.uuid4()) + '.jpg'  # Unique filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Make the prediction
    class_labels, r0, msg = preprocess_image(image_path)

    # Class names (adjust these based on your model)

    # Display the prediction result
    result = {
        'class_name': class_labels,
        'Confidence': r0,
        'Message':msg
    }

    return render_template('result.html', result=result, filename=filename)

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)