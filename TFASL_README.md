**ASL Alphabet Recognition with MobileNetV2**

This repository contains a deep learning project for recognizing American Sign Language (ASL) alphabets using the MobileNetV2 architecture fine-tuned on a custom dataset. The project includes data preprocessing, model training, evaluation, and test-time augmentation (TTA) for robust predictions.

Custom dataset link : https://www.kaggle.com/datasets/grassknoted/asl-alphabet

**Features**

- Dataset Handling: Automatically loads and preprocesses ASL alphabet images from a directory.
- Data Augmentation: Random transformations to improve model generalization.
- Model Architecture: Transfer learning with MobileNetV2, including L2 regularization and dropout for regularization.
- Evaluation: Confusion matrix, classification report, and visualization of predictions.
- Test-Time Augmentation (TTA): Robust predictions through augmented test samples.
- Model Export: Save the trained model in both .h5 and .keras formats.
 
**Requirements**

Install the necessary dependencies
- pip install tensorflow matplotlib numpy scikit-learn seaborn opencv-python pillow

**How to Use**
- Place the ASL alphabet training and test datasets in the appropriate directories. 

**Train the Model**
- Run the nitebook to train the MobileNetV2 model with data augmentation

**Evaluate the Model**
- The script evaluates the model on test images using: Confusion matrix, Classification report, Test-time augmentation (TTA)

**Visualize Predictions**
- The script visualizes predictions on a subset of test images with their predicted labels and confidence scores.

**Key Functions**

- Dataset Loading and Preprocessing
- Load dataset using image_dataset_from_directory.
- Split into training and validation sets.
- Normalize images and apply augmentation.

**Model Definition**

- Transfer learning with MobileNetV2.
- L2 regularization and dropout for improved generalization.
- Fine-tuning of the last few layers.

**Training**

- Adam optimizer with a learning rate of 0.0001.
- Early stopping and model checkpointing for efficient training.

**Evaluation**

- Generate confusion matrix and classification report.
- Test-time augmentation (TTA) for robust predictions.

**Results**

The training script generates:

- Loss and accuracy curves for training and validation.
- Confusion matrix heatmap.
- Classification report with precision, recall, and F1 scores.

**Model Saving**

The trained model is saved in both .h5 and .keras formats for future use:

- asl_mobilenet_classifier.h5
- asl_mobilenet_classifier.keras

**Test-Time Augmentation (TTA)**

- The script applies data augmentation during inference to improve prediction robustness. Predictions from multiple augmented versions of the same image are averaged.

**Visualization**

- Visualize a subset of test images with predicted labels and confidence scores using matplotlib.

**License**

- This project is licensed under the MIT License. See the LICENSE file for details.

**Contributing**

- Contributions are welcome! Please open an issue or submit a pull request for any suggestions or improvements.

**Contact**

- For any questions or feedback, please contact Badri Narayanan Ramesh at bramesh@usc.edu or Vivin Thiyagarajan at vthiyaga@usc.edu
