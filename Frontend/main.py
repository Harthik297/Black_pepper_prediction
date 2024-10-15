import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2 

leaf_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)

def model_prediction(test_image):
    model = tf.keras.models.load_model("../Saved_models/trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(500, 500))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


def is_leaf_image(image):
    img = Image.open(image).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Convert to HSV color space
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Define the range for green color
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])

    # Create a mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Calculate the percentage of green pixels
    green_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    # Check if the percentage of green pixels is above a threshold
    green_percentage = (green_pixels / total_pixels) * 100
    return green_percentage > 10  # Check if more than 10% of the image is green

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("Black Pepper Plant Disease Recognition System")
    image_path = "../Images/Background.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Black Pepper Plant Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        # Predict button
        if st.button("Predict"):
            if is_leaf_image(test_image):
                st.write("Our Prediction")
                result_index = model_prediction(test_image)
                # Reading Labels
                class_name = ['Healthy', 'Quick wilt', 'Slow wilt']
                st.success("Model is predicting it's a {}".format(class_name[result_index]))
            else:
                st.error("Error: Please upload a valid leaf image.")
