import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

def main():
    img = Image.open("test-tube.png")
    st.set_page_config(page_title="Urinity App", page_icon=img)

    st.write("# 🧪Urinity App")
    st.write("### Benedict’s Reagent Classifier")
    st.write("The Benedict's Reagent Classifier is a deep learning model that classifies images of Benedict's solution test tubes into different glucose concentration levels.")
    st.write("The Benedict's Reagent Classifier utilizes transfer learning techniques, specifically ResNet, MobileNet, InceptionV3, and VGG16, to automate and simplify the process of determining glucose levels. By analyzing images of Benedict's solution test tubes, the classifier predicts the concentration of glucose, offering a rapid and efficient method for analysis.")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "About Us"))

    if page == "Home":
        display_home()
    elif page == "About Us":
        display_about()

def display_home():
    @st.cache(allow_output_mutation=True)
    def load_model(model_name):
        if model_name == "VGG16":
            model = tf.keras.models.load_model('vgg16_urinalysis.hdf5')
        elif model_name == "ResNet50":
            model = tf.keras.models.load_model('resnet_urinalysis.hdf5')
        elif model_name == "InceptionV3":
            model = tf.keras.models.load_model('inceptionv3_urinalysis.hdf5')
        elif model_name == "MobileNet":
            model = tf.keras.models.load_model('mobilenet_urinalysis.hdf5')
        else:
            raise ValueError("Invalid model name")
        return model

    def import_and_predict(image_data, model):
        size=(160, 160)
        image = ImageOps.fit(image_data, size, Image.LANCZOS)
        image = np.asarray(image)
        image = image / 255.0
        img_reshape = np.reshape(image, (1, 160, 160, 3))
        prediction = model.predict(img_reshape)
        return prediction

    model_names = ["VGG16", "ResNet50", "InceptionV3", "MobileNet"]
    selected_model = st.selectbox("Select a model", model_names)

    model = load_model(selected_model)
    class_names = ["High (>2 g%)", "Moderate (1.5-2 g%)", "No reducing sugar (0 %g)", "Traceable (0.5-1 g%)"]

    file = st.file_uploader("Choose photo from computer", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "Reducing Sugar Level: " + class_name
        st.success(string)
        st.image(image, use_column_width=True)

def display_about():
    st.write("# About Us")

def toggle_theme():
    if st.button("Toggle Theme"):
        current_theme = st.get_theme()
        if current_theme == "light":
            new_theme = "dark"
        else:
            new_theme = "light"
        st.set_theme(new_theme)

if __name__ == "__main__":
    toggle_theme()
    main()
