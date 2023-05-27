import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

def main():
    st.title("Urinity App")

    @st.cache(allow_output_mutation=True)
    def load_model(model_name):
        if model_name == "VGG16":
            model = tf.keras.models.load_model('vgg16_urinalysis.hdf5')
        elif model_name == "ResNet50":
            model = model = tf.keras.models.load_model('resnet_urinalysis.hdf5')
        elif model_name == "InceptionV3":
            model = model = tf.keras.models.load_model('inceptionv3_urinalysis.hdf5')
        elif model_name == "MobileNet":
            model = model = tf.keras.models.load_model('mobilenet_urinalysis.hdf5')
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
    class_names = ["High", "Moderate", "No reducing sugar", "Traceable"]

    st.write("### Benedict’s Reagent Classifier")

    file = st.file_uploader("Choose photo from computer", type=["jpg", "png", "jpeg"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        string = "No Reducing Sugar Level: " + class_name
        st.success(string)

if __name__ == "__main__":
    main()
