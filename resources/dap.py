from types import SimpleNamespace
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
from PIL import Image, ImageEnhance
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2
#---------------------------------------------------------#
# F U N C T I O N S    F O R    F A C E    B L U R R I N G
#---------------------------------------------------------#

def anonymize_face_pixelate(image, blocks=3):
    # divide the input image into NxN blocks
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")

    # loop over the blocks in both the x and y direction
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original image
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (B, G, R), -1)

    # return the pixelated blurred image
    return image

def blur(image):
    # loading the image
    orig = image.copy()
    (h, w) = image.size[::-1]
    # st.write((h,w))
    img = np.array(image.convert('RGB'))
    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
    (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    # print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # detection
        confidence = detections[0, 0, i, 2]

    # filter out weak detections by ensuring the confidence is greater
    # than the minimum confidence
        if confidence > 0.4:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = img[startY:endY, startX:endX]

            face = anonymize_face_pixelate(face,
                blocks = 10) # higher no of blocks gives a more 'blurred' image

            # store the blurred face in the output image
            img[startY:endY, startX:endX] = face

    output = np.hstack([orig, img])
    return img, orig

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# F U N C T I O N S   F O R   G R A D - C A M 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
from pyimagesearch.gradcam import GradCAM

def use_gradcam(image, original_path):
    # initialize the model to be ResNet50
    Model = ResNet50
    # load the pre-trained CNN from disk
    print("[INFO] loading model...")
    model = Model(weights="imagenet")
    # load the original image from disk (in OpenCV format) and then resize the image to its target dimensions
    orig = image
    resized = cv2.resize(orig, (224,224))
    # load the input image from disk (in Keras/TensorFlow format) and preprocess it
    image = load_img(original_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    # using pre trained model to predict
    preds = model.predict(image)
    i = np.argmax(preds[0])

    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)

    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.25)
    output = imutils.resize(output)
    return output 
#--------------------------------------------------------#
st.set_page_config(page_title='Distracted driver detection',
                   layout='wide')
st.write('# **Distracted drivers detection**')
#---------------------------------------------------------#
# D O W N L O A D I N G necessary files
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(config = prototxtPath, model = weightsPath, framework = 'Caffe')
#st.write(net)
#---------------------------------------------------------#
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

st.write('This section blurs the face of the image that you are using and is useful to anonymize the driver. In addition, it shows the features that the model focuses on when making a prediction on the class of the driver.')

if uploaded_file is not None:
    img_path = uploaded_file.name
    st.write('### **This is your original image**')
    image = Image.open(uploaded_file)
    st.image(image)
    # BLURRING THE IMAGE  
    st.write('### **This is your new image**')   
    blurred_img = blur(image)[0]
    st.image(blurred_img)

    # G R A D    C A M 
    st.write('### **This is the class activation map of the model**')
    st.image('color map details.png')
    st.image(use_gradcam(blurred_img, img_path))