import os
import cv2
import tensorflow as tf
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.graphics.texture import Texture
from kivy.config import Config
from kivy.clock import Clock
import numpy as np

class SplashScreen(Screen):
    def __init__(self, **kwargs):
        super(SplashScreen, self).__init__(**kwargs)
        image_widget = Image(source=self.get_image_path('splash_screen.png'))
        self.add_widget(image_widget)

    def get_image_path(self, filename):
        return os.path.join(os.path.dirname(__file__), 'images', filename)

class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super(CameraScreen, self).__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.ids.image_widget = Image()
        self.label_widget = Label(text='', size_hint=(1, None), height='48dp')
        capture_button = Button(text="Capture", size_hint=(1, None), height='48dp')
        capture_button.bind(on_press=self.capture_image)
        save_button = Button(text="Save", size_hint=(1, None), height='48dp')
        save_button.bind(on_press=self.save_image)
        layout.add_widget(self.ids.image_widget)
        layout.add_widget(self.label_widget)
        layout.add_widget(capture_button)
        layout.add_widget(save_button)
        self.add_widget(layout)

        # Load your pre-trained deep learning model
        self.model = tf.keras.models.load_model(self.get_model_path('flower_classifier.hdf5'))

    def capture_image(self, instance):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        self.ids.image_widget.texture = texture
        self.captured_image = frame
        cap.release()

    def save_image(self, instance):
        if hasattr(self, 'captured_image'):
            cv2.imwrite(self.get_image_path('captured_image.png'), self.captured_image)
            print("Image saved successfully.")
            self.perform_inference(self.captured_image)
        else:
            print("No image has been captured.")
    
    def get_model_path(self, filename):
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, filename)
        return model_path
    
    def get_image_path(self, filename):
        # Get the absolute path to the image file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, filename)
        return image_path

    def perform_inference(self, image):
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        result = self.process_prediction(prediction)
        print(result)
        self.label_widget.text = result

    def preprocess_image(self, image):
        # Preprocess the image (e.g., resize, normalization) for your deep learning model
        processed_image = cv2.resize(image, (128, 128))  # Example: Resize to (128, 128)
        processed_image = np.expand_dims(processed_image, axis=0)
        processed_image = processed_image / 255.0  # Example: Normalize pixel values between 0 and 1
        return processed_image

    def process_prediction(self, prediction):
        # Process the prediction result and return the predicted class label
        class_labels = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = class_labels[predicted_class_index]
        return predicted_class_label
    
class CameraApp(App):
    def build(self):
        screen_manager = ScreenManager()
        splash_screen = SplashScreen(name='splash')
        camera_screen = CameraScreen(name='camera')
        screen_manager.add_widget(splash_screen)
        screen_manager.add_widget(camera_screen)
        screen_manager.current = 'splash'
        Clock.schedule_once(lambda dt: setattr(screen_manager, 'current', 'camera'), 2)
        return screen_manager

    def on_stop(self):
        # Clean up any resources before the app exits
        # For example, release the deep learning model if necessary
        pass

if __name__ == '__main__':
    # Set the width and height of the app window
    window_width = 360
    window_height = 640

    # Adjust the window size based on the device's screen size
    screen_width = int(Config.get('graphics', 'width'))
    screen_height = int(Config.get('graphics', 'height'))
    scale_factor = min(screen_width / window_width, screen_height / window_height)
    adjusted_width = int(window_width * scale_factor)
    adjusted_height = int(window_height * scale_factor)
    Config.set('graphics', 'width', str(adjusted_width))
    Config.set('graphics', 'height', str(adjusted_height))

    # Run the app
    CameraApp().run()

