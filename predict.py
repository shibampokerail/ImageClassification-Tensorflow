import tensorflow as tf
from train import model
import cv2
import sys
import os

CATEGORIES = ["Boys_Apparel", "Girls_Apparel", "Mens_Footwear", "Womens_Footwear"]

def convert_input_image(path):
    image_array = cv2.imread(path)
    img_shape = cv2.resize(image_array, (256, 256))
    return img_shape.reshape(1, 256, 256, 3)

model = tf.keras.models.load_model("object_detection_v1.h5")

def predict_image(image_name):
    prediction = model.predict([convert_input_image(image_name)]).round()
    print(prediction)
    if prediction[0][0]>=1:
        return CATEGORIES[0]
    if prediction[0][1]>=1:
        return CATEGORIES[1]
    if prediction[0][2]>=1:
        return CATEGORIES[2]
    if prediction[0][3]>=1:
        return CATEGORIES[3]

GOOGLE_IMAGES_DIR = 'google_images_for_external_testing\\'

files = os.listdir(GOOGLE_IMAGES_DIR)

googled_images = [file for file in files if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))]

if len(sys.argv) < 2:
    for image in googled_images:
        print(image, "====> ", predict_image(GOOGLE_IMAGES_DIR + image))
else:
    print(sys.argv[1], "====> ", predict_image(sys.argv[1]))



