import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from deepforest import main
from deepforest import get_data
model = main.deepforest()
model.use_release()
def count(img):
    sample_image_path = get_data(img)
    cnt = model.predict_tile(img, return_plot = True,iou_threshold=0.4, patch_size=250)
    return cnt
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
        imagrArr = count(original_image)
        im = Image.fromarray(imageArr)
        count_of_trees = len(imageArr)
        # Save modified image
        save_image(im,"count.jpg")
        # Display saved image
        saved_image = Image.open('count.jpg')
        st.image(saved_image, caption='e', use_column_width=True)

if __name__ == '__main__':
    main()
