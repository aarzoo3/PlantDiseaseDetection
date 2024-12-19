import tensorflow as tf
import numpy as np
import json

#cnn model prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model(r'C:\Users\aarzoov\PlantDiseaseDetection\trained_plant_disease_model_10epoch.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    with open(r"C:\Users\aarzoov\PlantDiseaseDetection\model_classes.json", 'r') as json_file:
        classes = json.load(json_file)
    class_mapping = {idx: cls for idx, cls in enumerate(classes)}


    return class_mapping[np.argmax(predictions)]


x = model_prediction(r"C:\Users\lenovo\OneDrive\Desktop\Aarzoo_photo_1_11zon.jpg")
print("The Predictions is -", x)