import streamlit as st
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm

import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt

# Function definitions
weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
# Define the custom loss function
def dice_loss_plus_1focal_loss(y_true, y_pred):
   dice_loss = sm.losses.DiceLoss(class_weights = weights)
   focal_loss = sm.losses.CategoricalFocalLoss()
   total_loss = dice_loss + (1 * focal_loss)
   return total_loss

def jaccard_coef(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return final_coef_value

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess the input image."""
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

def predict_mask(model, image):
    """Perform predictions on the input image."""
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    return predicted_mask

def save_masked_image(mask, output_path):
    """Save the predicted mask image."""
    plt.figure(figsize=(mask.shape[1]/100, mask.shape[0]/100))  # Adjust the figure size based on mask dimensions
    plt.imshow(mask)  # Ensure the mask is displayed in color
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Ensure no whitespace around the saved image
    st.image(output_path)

def calculate_class_percentage(mask_image_path):
    # Load the masked image
    mask_image = cv2.imread(mask_image_path)
    mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    
    # Convert RGB image to HSV color space
    hsv_image = cv2.cvtColor(mask_image_rgb, cv2.COLOR_RGB2HSV)
    
    # Define the HSV ranges for each class
    class_ranges = {
        'building': ((110, 50, 50), (130, 255, 255)),   # Blue color for buildings
        'land': ((130, 50, 50),  (170, 255, 255)),      # Updated HSV range for land
        'road': ((90, 50, 50), (110, 255, 255)), 
        'water': ((10, 10, 10),  (21, 255, 255)),     # Updated HSV range for roads
        'vegetation': ((22, 10, 10),  (38, 255, 255)),  # Updated HSV range for vegetation
    }
    
    class_masks = {}
    
    # Iterate over each class
    for class_name, (lower_range, upper_range) in class_ranges.items():
        # Create a mask for the class based on its HSV range
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        class_masks[class_name] = mask
    
    # Calculate the area of each class mask
    class_areas = {class_name: np.sum(mask > 0) for class_name, mask in class_masks.items()}
    
    # Calculate the total area of the image
    total_area = mask_image.shape[0] * mask_image.shape[1]
    
    # Calculate the percentage of area for each class
    class_percentages = {class_name: (area / total_area) * 100 for class_name, area in class_areas.items()}
    
    return mask_image_rgb, class_masks, class_percentages

# Load the model with custom loss and metric functions
model_path = "satellite_segmentation_full.h5"
model = load_model(model_path, custom_objects={'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss, 'jaccard_coef': jaccard_coef})

def main():
    st.title("Satellite Image Segmentation")

    uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Satellite Image.', use_column_width=True)

        # Preprocess the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed_image = preprocess_image(image)

        # Perform prediction
        predicted_mask = predict_mask(model, processed_image)

        # Save and display the predicted mask
        output_path = "predicted_mask.png"
        save_masked_image(predicted_mask, output_path)

        # Calculate class percentages
        mask_image_path = output_path
        mask_image, class_masks, class_percentages = calculate_class_percentage(mask_image_path)

        # Display class masks and percentages
        st.subheader("Class Masks and Percentages")
        visualize_class_masks(mask_image, class_masks, class_percentages)

if __name__ == "__main__":
    main()
