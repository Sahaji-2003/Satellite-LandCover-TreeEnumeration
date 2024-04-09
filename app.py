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
    # st.image(output_path)

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

def visualize_predictions(image, canny_image, mask):
    """Visualize the original image, Canny image, and predicted mask."""
    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Display the Canny image
    st.image(canny_image, caption='Canny Image', use_column_width=True, channels='GRAY')

    # Create a plot for the predicted mask
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask)
    ax.set_title("Predicted Mask")
    ax.axis('off')

    # Display the plot using Streamlit
    st.pyplot(fig)


def visualize_class_masks(mask_image, class_masks, class_percentages):
    # Display the original image
    st.image(mask_image, caption='Original Image', use_column_width=True)

    # Display the class masks
    for class_name, mask in class_masks.items():
        if class_name != 'unlabeled':
            st.image(mask, caption=f"{class_name.capitalize()} Mask", use_column_width=True, channels='GRAY')

            # Print percentage value in the console
            st.write(f"{class_name.capitalize()}: {class_percentages[class_name]:.2f}%")


def apply_canny(image_path, threshold1=50, threshold2=150, target_size=(256, 256)):
    # Read the original image
    original_image = image_path

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(blurred_image, threshold1, threshold2)

    # Resize the canny image to the target size
    canny_resized = cv2.resize(canny_edges, target_size)

    return canny_resized

# Load the model with custom loss and metric functions
model_path = "satellite_segmentation_full.h5"
model = load_model(model_path, custom_objects={'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss, 'jaccard_coef': jaccard_coef})

# def main():
# # Create a navigation bar on the left side
#     st.sidebar.title('Navigation')

# # Add links to different sections of the app
#     selection = st.sidebar.radio("Go to", ['Home', 'About', 'Contact'])

# # Define the content for each section
#     if selection == 'Home':
#       st.title('Home Page')
#       st.write('Welcome to the Home Page!')
#     elif selection == 'About':
#       st.title('About Page')
#       st.write('This is the About Page. It provides information about the app.')
#     elif selection == 'Contact':
#       st.title('Contact Page')
#       st.write('You can contact us at contact@example.com.')

#     st.title("Satellite Image Segmentation")

#     uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#         st.image(uploaded_file, caption='Uploaded Satellite Image.', use_column_width=True)

#         # Preprocess the uploaded image
#         image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
#         processed_image = preprocess_image(image)

#         # Perform prediction
#         predicted_mask = predict_mask(model, processed_image)

#         # Save and display the predicted mask
#         output_path = "predicted_mask.png"
#         save_masked_image(predicted_mask, output_path)

#         # Calculate class percentages
#         mask_image_path = output_path
#         mask_image, class_masks, class_percentages = calculate_class_percentage(mask_image_path)

#         st.subheader("Masked and Canny Images")
#         cols = st.columns(2)
#         with cols[0]:
#             st.image(mask_image, caption='Predicted Mask', use_column_width=True)
#         with cols[1]:
#             canny_edges = apply_canny(image)
#             st.image(canny_edges, caption='Canny Image', use_column_width=True, channels='GRAY')

#         # Display class percentages
#         st.subheader("Class Distribution")
#         for class_name, percentage in class_percentages.items():
#             st.write(f"{class_name.capitalize()}: {percentage:.2f}%")

#         # Display class masks and percentages
#         # Display class masks and percentages in a similar format to "Original, Predicted, and Canny Images"
#         st.subheader("Class Masks and Percentages")
#         cols = st.columns(3)

#         # Display class masks
#         for class_name, mask in class_masks.items():
#             if class_name != 'unlabeled':
#                with cols[0]:
#                   st.image(mask, caption=f"{class_name.capitalize()} Mask", use_column_width=True, channels='GRAY')
#                with cols[1]:
#                   st.write("")
#                with cols[2]:
#                   st.write("")
 
#         # Display percentages
#         for class_name, percentage in class_percentages.items():
#             if class_name != 'unlabeled':
#                with cols[0]:
#                  st.write("")
#                with cols[1]:
#                  st.write(f"{class_name.capitalize()}: {percentage:.2f}%")
#                with cols[2]:
#                  st.write("")
 

# if __name__ == "__main__":
#     main()






def main():
    # Stylish header with navigation bar
    st.markdown(
        """
        <style>
        .header {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            width: 100%; /* Make the header cover the full width */
        }
        .header h1 {
            color: #333333;
            text-align: center;
            font-size: 36px;
            margin: 0; /* Remove any default margins */
        }
        .nav {
            display: flex;
            align-items: center;
        }
        .nav a {
            margin-right: 20px;
            color: #333333;
            text-decoration: none;
        }
        .nav a:hover {
            text-decoration: underline;
        }
        </style>
        """
    , unsafe_allow_html=True)

    # Header content with navigation links
    st.markdown(
        """
        <div class="header">
            <h1>Satellite Image Segmentation</h1>
            <div class="nav">
                <a href="#masked_canny">Masked and Canny Images</a>
                <a href="#class_distribution">Class Distribution</a>
                <a href="#all_class_masks">All Class Masks</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # st.title("Satellite Image Segmentation")

    uploaded_file = st.file_uploader("Upload Satellite or Drone Image", type=["jpg", "png", "jpeg"])


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

        # Display the main content
        st.subheader("Masked and Canny Images")
        st.markdown('<a id="masked_canny"></a>', unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.image(mask_image, caption='Predicted Mask', use_column_width=True)
        with cols[1]:
            canny_edges = apply_canny(image)
            st.image(canny_edges, caption='Canny Image', use_column_width=True, channels='GRAY')

        # Display class percentages and graph side by side
        st.subheader("Class Distribution")
        st.markdown('<a id="class_distribution"></a>', unsafe_allow_html=True)
        cols_distribution = st.columns(2)
                

        # Display class percentages
        with cols_distribution[0]:
            st.write("Class Percentages:")
            for class_name, percentage in class_percentages.items():
                st.write(f"{class_name.capitalize()}: {percentage:.2f}%")

        # Display bar graph
        with cols_distribution[1]:
            st.write("Class Distribution Graph:")
            labels = list(class_percentages.keys())
            sizes = list(class_percentages.values())
            colors = [plt.cm.tab10(i/float(len(labels))) for i in range(len(labels))]  # Generate different colors
            fig, ax = plt.subplots()
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, sizes, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()  # Invert y-axis to display the highest bar at the top
            ax.set_xlabel('Percentage')
            ax.set_title('Class Distribution')
            st.pyplot(fig)

        # Display pie chart in the sidebar
        st.sidebar.title('Class Distribution')
        for class_name, percentage in class_percentages.items():
            st.sidebar.write(f"{class_name.capitalize()}: {percentage:.2f}%")

        # Display pie chart in the sidebar
        labels = list(class_percentages.keys())
        sizes = list(class_percentages.values())
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax1.axis('equal')

        st.sidebar.pyplot(fig1)            
        st.sidebar.write("")
        st.sidebar.write("Jump to:")
        st.sidebar.write("[Masked and Canny Images](#masked_canny)")
        st.sidebar.write("[Class Distribution](#class_distribution)")
        st.sidebar.write("[All Class Masks](#all_class_masks)")
  


        # Display all class mask images
        st.subheader("All Class Masks")
        st.markdown('<a id="all_class_masks"></a>', unsafe_allow_html=True)
        cols_masks = st.columns(2)  # Display 2 images per row
        for i, (class_name, mask) in enumerate(class_masks.items()):
            if class_name != 'unlabeled':
                if i % 2 == 0:
                    mask_column = cols_masks[0]
                else:
                    mask_column = cols_masks[1]
                with mask_column:
                    st.image(mask, caption=f"{class_name.capitalize()} Mask", use_column_width=True, channels='GRAY')

if __name__ == "__main__":
    main()





