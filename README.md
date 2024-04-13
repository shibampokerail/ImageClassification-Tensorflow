## Image Classification with tensorflow
Classifies the following images in the folder(imgDataSet):
- Boys Apparel
- Girls Apparel
- Mens footwear
- Womens footwear

## Setting Up your environment
```
pip install -r requirements.txt
```

The two libraries used are tensorflow and opencv. So if you are trying to run this from google colab or jupyter notebook then just:
```
pip install tensorflow
pip install opencv-python
```

## Training the model 
The current model "object_detection_v1.h5" has an accuracy of 81% and is trained only for 12 epochs using train.py . You can train your own model by changing the layers and nodes in train.py.
```
python train.py
```

## Predicting from the model
There are two ways to run the prediction script.
- This will predict all the images in the "google_images_for_external_testing" folder you can add more images into that folder.
```
python predict.py 
```
- This will predict any image you give it.
```
python predict.py <path_of_your_custom_image>
```
