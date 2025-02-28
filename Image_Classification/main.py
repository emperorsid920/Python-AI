# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your trained model
model = load_model('/Users/sidkumar/Documents/Portfolio Freelance/Image_Classification/cifar10_model_augmented.h5')

# Define a function for model prediction
def predict(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def make_prediction():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file")

    if file:
        # Save the file to the 'uploads' directory
        file_path = 'uploads/' + secure_filename(file.filename)
        file.save(file_path)

        # Make prediction
        predictions = predict(file_path)

        # Get the class with the highest probability
        class_index = np.argmax(predictions[0])

        # Example: Map class indices to class labels (replace with your class labels)
        class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck","tiger"]
        class_label = class_labels[class_index]

        # Get the probability of the predicted class
        probability = predictions[0][class_index]

        # Format the prediction text
        prediction_text = f"Predicted Class: {class_label} (Probability: {probability:.2f})"

        return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
