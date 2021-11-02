from types import SimpleNamespace
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import argparse
# import imutils
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
# import imutils
import cv2
#---------------------------------------------------------#
# F U N C T I O N S    F O R    F A C E    B L U R R I N G
#---------------------------------------------------------#


def app():
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

                face = anonymize_face_pixelate(face, blocks=10)
                # higher no of blocks gives a more 'blurred' image
                # store the blurred face in the output image
                img[startY:endY, startX:endX] = face

        output = np.hstack([orig, img])
        return img, orig

    #--------------------------------------------------------#
    # st.set_page_config(layout='wide') #optional to make the layout widescreen
    st.title('Face Blurring')
    st.header('Upload an Image to see it become blurred!')
    #---------------------------------------------------------#
    # D O W N L O A D I N G necessary files
    prototxtPath = "./resources/deploy.prototxt"
    weightsPath = "./resources/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNet(config=prototxtPath,
                          model=weightsPath, framework='Caffe')
    # st.write(net)
    #---------------------------------------------------------#

    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        st.write('### **This is your original image**')
        image = Image.open(uploaded_file)
        st.image(image)
        # BLURRING THE IMAGE
        st.write('### **This is your new image**')
        blurred_img = blur(image)[0]
        st.image(blurred_img, use_column_width=True)
