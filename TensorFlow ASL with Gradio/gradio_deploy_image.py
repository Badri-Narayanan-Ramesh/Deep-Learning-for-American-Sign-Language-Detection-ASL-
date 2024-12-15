import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image as PILImage

# Load the trained ASL recognition model (ensure the file path is correct)
asl_gesture_model = tf.keras.models.load_model("asl_mobilenet_classifier_20241215_014404.h5")

# Define gesture labels for ASL
asl_gesture_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'
]

# Function to create a bar chart visualizing the confidence of predictions
def create_confidence_visual(predictions):
    plt.figure(figsize=(8, 4))
    plt.bar(asl_gesture_labels, predictions[0])
    plt.xticks(rotation=90)
    plt.ylabel('Confidence Percentage')
    plt.title('Confidence Distribution for Predicted ASL Gesture')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Convert to PIL image for Gradio interface
    chart_image = PILImage.open(buf)
    return chart_image

# Function to classify ASL gestures from the input image and provide the results
def predict_gesture_from_image(image, image_name):
    # Prepare the image by converting to RGB and resizing
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Resize to match the model's input size
    image_array = np.array(image)  # Convert image to numpy array
    image_array = image_array / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension for prediction
    
    # Model makes a prediction
    prediction_scores = asl_gesture_model.predict(image_array)
    predicted_gesture = asl_gesture_labels[np.argmax(prediction_scores)]  # Get the most likely gesture
    confidence_percentage = np.max(prediction_scores) * 100  # Get confidence percentage
    
    # If confidence is low, return uncertain prediction
    if confidence_percentage < 50:
        predicted_gesture = f"Prediction: Uncertain (Confidence: {confidence_percentage:.2f}%)"
    else:
        predicted_gesture = f"Prediction: {predicted_gesture} (Confidence: {confidence_percentage:.2f}%)"
    
    # Handle case when image name is not provided
    if not image_name:
        image_name = "Unnamed Image"
    
    # Generate a bar chart for the confidence distribution
    confidence_chart = create_confidence_visual(prediction_scores)
    
    # Return the prediction text and confidence chart image
    return predicted_gesture, confidence_chart

# Setting up the Gradio interface with user-friendly features
gesture_recognition_app = gr.Interface(
    fn=predict_gesture_from_image,
    inputs=[gr.Image(type="pil", label="Upload Gesture Image"), gr.Textbox(label="Image Identifier (optional)", placeholder="Enter an optional name for the image")],
    outputs=[gr.Textbox(label="Prediction Results"), gr.Image(label="Confidence Distribution Chart")],
    title="ASL Gesture Classification",
    description="Upload an image of an ASL gesture to classify it using a trained model. The result will include the predicted gesture, confidence score, and a chart displaying prediction confidence.",
    theme="gradio",  # Use a clean theme
    live=True,
)

# Launch the application
gesture_recognition_app.launch(share=True)