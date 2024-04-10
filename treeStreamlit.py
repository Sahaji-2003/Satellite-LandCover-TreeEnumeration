import streamlit as st
from PIL import Image
import numpy as np

def save_image(image, filename):
    image.save(filename)

def main():
    st.title('Tree Count')

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display original image
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption='Original Image', use_column_width=True)
        tree
        # Convert image to grayscale
        grayscale_image = to_grayscale(original_image)
        st.image(grayscale_image, caption='Grayscale Image', use_column_width=True)

        # Save modified image
        save_image(grayscale_image, 'grayscale_image.jpg')

        # Display saved image
        saved_image = Image.open('grayscale_image.jpg')
        st.image(saved_image, caption='Saved Image', use_column_width=True)

if __name__ == '__main__':
    main()
