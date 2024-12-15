import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image as PILImage
import cv2

# Load the trained ASL model (ensure the file path is correct)
asl_recognition_model = tf.keras.models.load_model("asl_mobilenet_classifier_20241215_014404.h5")

# Define the alphabet labels for ASL gestures
asl_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing'
]

# Function to generate a bar chart for the ASL prediction confidence
def create_confidence_chart(predictions):
    plt.figure(figsize=(8, 4))
    plt.bar(asl_labels, predictions[0])
    plt.xticks(rotation=90)
    plt.ylabel('Confidence')
    plt.title('Confidence Scores for ASL Gesture Recognition')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    
    # Convert the chart to a PIL image for Gradio display
    chart_image = PILImage.open(buf)
    return chart_image

# Function to detect the object (hand gesture) and make predictions
def detect_and_predict(frame):
    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize frame to match the input size for the model (224x224)
    resized_frame = cv2.resize(frame_rgb, (224, 224))

    # Normalize and prepare the image for prediction
    image_array = np.array(resized_frame) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict the gesture using the ASL model
    predictions = asl_recognition_model.predict(image_array)

    # Extract the predicted class and confidence score
    predicted_gesture = asl_labels[np.argmax(predictions)]
    confidence_score = np.max(predictions) * 100

    # If the confidence is low, display "Uncertain"
    if confidence_score < 50:
        predicted_gesture = f"Uncertain (Confidence: {confidence_score:.2f}%)"
    else:
        predicted_gesture = f"{predicted_gesture} (Confidence: {confidence_score:.2f}%)"

    # Annotate the frame with the predicted class (gesture)
    cv2.putText(frame_rgb, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert frame back to PIL image for Gradio
    final_frame = PILImage.fromarray(frame_rgb)

    return final_frame, predicted_gesture, predictions

# Function to handle image input
def process_image(image):
    processed_image, prediction, predictions = detect_and_predict(image)
    confidence_chart = create_confidence_chart(predictions)
    return processed_image, prediction, confidence_chart

# Function to handle video input (real-time gesture recognition)
def process_video_frame(video):
    # Process the video frame and get predictions
    processed_frame, prediction, _ = detect_and_predict(video)
    return processed_frame, prediction

# Set up the Gradio interface for image input
image_interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),  # Image input
    outputs=["image", "text", "image"],  # Image output with prediction and confidence chart
    live=True,
    title="Interactive ASL Gesture Recognition with Image Input",
    description="Upload an image to identify ASL gestures and view confidence scores."
)

# Set up the Gradio interface for video input (real-time recognition)
video_interface = gr.Interface(
    fn=process_video_frame,
    inputs=gr.Video(),  # Video input
    outputs=["image", "text"],  # Real-time video output with predictions
    live=True,
    title="Real-Time ASL Gesture Recognition with Video Input",
    description="Use your webcam to identify ASL gestures in real-time."
)

# Launch the interfaces
image_interface.launch(share=True)  # Image app
video_interface.launch(share=True)  # Video app