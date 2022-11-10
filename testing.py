import os
import glob
from tensorflow import keras
import cv2 as cv
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import re
import argparse

# classes among which the image will be differentiated
Classes = ["Apple___Apple_scab", 
           "Apple___Black_rot", 
           "Apple___Cedar_apple_rust", 
           "Apple___healthy", 
           "Blueberry___healthy", 
           "Cherry_(including_sour)___Powdery_mildew", 
           "Cherry_(including_sour)___healthy",
           "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", 
           "Corn_(maize)___Common_rust",  
           "Corn_(maize)___Northern_Leaf_Blight", 
           "Corn_(maize)___healthy",
           "Grape___Black_rot", 
           "Grape___Esca_(Black_Measles)", 
           "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",  
           "Grape___healthy",
           "Orange___Haunglongbing_(Citrus_greening)", 
           "Peach___Bacterial_spot", 
           "Peach___healthy", 
           "Pepper__bell___Bacterial_spot", 
           "Pepper__bell___healthy", 
           "Potato___Early_blight",
           "Potato___Late_blight",
           "Potato___healthy",
           "Raspberry___healthy", 
           "Soybean___healthy", 
           "Squash___Powdery_mildew", 
           "Strawberry___Leaf_scorch",
           "Strawberry___healthy",
           "Tomato___Bacterial_spot",
           "Tomato___Early_blight",
           "Tomato___Late_blight",
           "Tomato___Leaf_Mold",
           "Tomato___Septoria_leaf_spot",
           "Tomato___Spider_mites Two-spotted_spider_mite",
           "Tomato___Target_Spot",
           "Tomato___Tomato_Yellow_Leaf_Curl_Virus" ,
           "Tomato___Tomato_mosaic_virus",
           "Tomato___healthy",
           ]

# As test images have some different naming convention as compared to the classes we have, we are mapping back the image name to their original class name convention. If you have test images with class name convention, no need to define this step.
map_classes_dict = {
    'AppleCedarRust' : 'Apple___Cedar_apple_rust',
    'AppleScab' : 'Apple___Apple_scab',
    'CornCommonRust' : 'Corn_(maize)___Common_rust',
    'PotatoEarlyBlight' : 'Potato___Early_blight',
    'PotatoHealthy' : 'Potato___healthy',
    'TomatoEarlyBlight' : 'Tomato___Early_blight',
    'TomatoHealthy' : 'Tomato___healthy',
    'TomatoYellowCurlVirus' : 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
}

# preparing image for prediction
def prepare_for_prediction(img_path):
    img_cv = cv.imread(img_path)
    # converting to RGB format as Keras requires RGB format
    img_cv = cv.cvtColor(img_cv,cv.COLOR_BGR2RGB)
    img = img_cv/255.0
    return np.expand_dims(img, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",required=True,help="Path to test dataset")
    args = parser.parse_args()
    # loading the model 
    model = keras.models.load_model(r'D:\Machine_Learning\Plant_Diseases\iteration_3_25_epochs\model_1.h5')
    # loading model weights
    model.load_weights(r'D:\Machine_Learning\Plant_Diseases\iteration_3_25_epochs\model00000008.h5')
    # test data
    test_data = args.test
    # variable to store the number of correct class predictions
    correct_class_predictions = 0
    # variable to store the number of images tested
    number_of_test_images = 0
    for image_path in glob.glob(test_data + "/" + "*.JPG"):
        number_of_test_images += 1
        # image_name without extension
        image_name = image_path.split("\\")[6].split(".")[0]
        check = any(chr.isdigit() for chr in image_name)
        if check:
            temp = re.compile("([a-zA-Z]+)([0-9]+)")
            # grouping numbers and letters seperatly 
            image_name = image_name.split("[0-9")[0]
            res = (temp.match(image_name).groups())[0]
            actual_class = map_classes_dict[res]
        else:
            actual_class = image_name
        # model prediction
        predict = model.predict([prepare_for_prediction(image_path)])
        classes_x=np.argmax(predict,axis=1)
        print('Prediction data',actual_class,Classes[int(classes_x)])
        if(Classes[int(classes_x)] == actual_class):
            correct_class_predictions += 1
    # class_prediction_accuracy
    class_prediction_accuracy = round(((correct_class_predictions*100)/number_of_test_images),2)
    print("Weight Class Prediction Accuracy - " + str(class_prediction_accuracy))

      



















        



