import tensorflow as tf
from train import model
import cv2

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

googled_images = ["girlsapparel.jpg","boysapparel.jpg","mensshoes.jpg","heels.png"]
GOOGLE_IMAGES_DIR = 'google_images_for_external_testing\\'

for image in googled_images:
    print(image, "====> ", predict_image(GOOGLE_IMAGES_DIR + image))


