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
    st.write("The Benedict's Reagent Classifier utilizes transfer learning techniques, specifically MobileNet, ResNet, InceptionV3, and VGG16, to automate and simplify the process of determining glucose levels. Among these models, MobileNet demonstrates exceptional accuracy in analyzing images of Benedict's solution test tubes and predicting the concentration of glucose, offering a rapid and efficient method for analysis.")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "About Us"))

    if page == "Home":
        display_home()
    elif page == "About Us":
        display_about()

def display_home():
    @st.cache(allow_output_mutation=True)
    def load_model(model_name):
        if model_name == "MobileNet":
            model = tf.keras.models.load_model('mobilenet_urinalysis.hdf5')
        elif model_name == "VGG16":
            model = tf.keras.models.load_model('vgg16_urinalysis.hdf5')
        elif model_name == "ResNet50":
            model = tf.keras.models.load_model('resnet_urinalysis.hdf5')
        elif model_name == "InceptionV3":
            model = tf.keras.models.load_model('inceptionv3_urinalysis.hdf5')
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

    model_names = ["MobileNet", "ResNet50", "InceptionV3", "VGG16"]
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
    
    st.write("### Selwyn Landayan")
    st.write("##### Student, Technological Institute of the Philippines")
    st.write("Selwyn Landayan's passion for technology and desire to exert a positive influence on the field of computer engineering are what motivate him. He has a strong foundation of understanding and a constant desire to learn, which makes him willing to take on new challenges and participate in innovative endeavors.")
    
    st.empty()
    st.write("### Jerome Marbebe")
    st.write("##### Student, Technological Institute of the Philippines")
    st.write("Jerome Marbebe is a dedicated computer engineering student with a profound passion for programming, machine learning, deep learning, and computer networking. He possesses a strong foundation in computer science and is driven by his curiosity to explore the intersection of these dynamic fields.")
    
    st.empty()
    st.write("### Sam Ryan Ruiz")
    st.write("##### Student, Technological Institute of the Philippines")
    st.write("Sam Ryan Ruiz is a highly motivated computer engineering student with a strong interest in technology and problem-solving. He has strong analytical abilities and a keen eye for detail, which enable him to excel at complex programming tasks. He thrives in collaborative settings, actively contributing ideas and working with peers to find creative solutions.")
    
    st.empty()
    st.write("### Airo Cornillez")
    st.write("##### Student, Technological Institute of the Philippines")
    st.write("Airo Craven Cornillez. He is a student from the Technological Institute of the Philippines - Quezon City, taking a Bachelor of Science in Computer Engineering. He graduated from Senior High School at Our Lady of Fatima University - Antipolo Campus with a strand in Science, Technology, Engineering, and Mathematics (STEM).")
if __name__ == "__main__":
    main()
