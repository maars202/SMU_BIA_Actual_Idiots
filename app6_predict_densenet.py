# https://docs.streamlit.io/en/stable/getting_started.html

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

def app():

    #importing the libraries
    # import streamlit as st
    import joblib
    from PIL import Image
    # from skimage.transform import resize
    import numpy as np
    import time
    import tensorflow as tf
    from cv2.cv2 import resize

    #Pre-trained model used to extract features from the images 
    #Removing top lets us use it for our own classification purposes
    import tensorflow
    from tensorflow.keras import layers, models
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
    from tensorflow.keras.applications import EfficientNetB3, EfficientNetB0, EfficientNetB7
    from tensorflow.keras.applications import DenseNet201
    from tensorflow.keras import Model
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_input_effnet
    from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense

    def create_densenet_m():

        baseModel = DenseNet201(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)), pooling = 'avg')
    #     baseModel = base

    #     baseModel = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        
        # construct the head of the model that will be placed on top of the
        # the base model
        # Freeze the base_model
        baseModel.trainable = False
    #     headModel = baseModel(inputs, training = False) 
        base_output = baseModel.output
        
    #     headModel = layers.Conv2D(filters = 32, activation = 'relu', kernel_size = (3,3))(headModel)
        x = layers.BatchNormalization(axis=1)

    # densenet output has 1920 units:
        x = layers.Dense(192, activation = 'relu')(base_output)
        x = layers.Dense(192, activation = 'relu')(base_output)
    #     x = layers.BatchNormalization(axis=1)(x)
        x = layers.Dense(10, activation = 'softmax')(x)

        model = Model(inputs=baseModel.input, outputs=x)
        return model





    model_mobile = create_densenet_m()
    mobile_weights_path = "./saved_weights/densenet201_5.h5 2"
    model_mobile.load_weights(mobile_weights_path)
    model_mobile.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', #data generator has encoded the classes already, so in vector form
        metrics = ['accuracy']
    )


    from sklearn.metrics import classification_report 
    from sklearn.metrics import confusion_matrix 
    def predict(test_images):
        print("predictingggg: ")
        predictions = model_mobile.predict(test_images)
        
        Y_prediction = np.argmax(predictions, axis =1 ) #Gets index of class with highest predicted pr
        print("prediction: ", Y_prediction, predictions)
        return Y_prediction, predictions

    # Designing the interface
    st.title("Distracted Driver Classification")
    # For newline
    st.write('\n')

    image = Image.open('./resources/image.jpeg')
    show = st.image(image, use_column_width=True)

    st.title("Upload Image")

    #Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Choose your own image
    uploaded_file = st.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )


    import tensorflow as tf
    d = {'filepaths': ["./images/image.jpeg"]}
    df = pd.DataFrame(data=d)

    def generate_data(processor, test_df = df):
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function = processor
        )

        test_images = generator.flow_from_dataframe(
            dataframe = test_df,
            x_col = 'filepaths', 
            # y_col = 'Label',
            target_size = (224, 224), #Default for 
            color_mode = 'rgb', 
            class_mode = 'categorical',
            batch_size = 24
            # shuffle = False
            # seed = 1
        )

        
        return test_images

    def process_img(img_array):
        print("process_img running")
        img_array.resize((224, 224))
        img = np.expand_dims(img_array,axis = 0)
        return img




    if uploaded_file is not None:
        
        u_img = Image.open(uploaded_file)
        show.image(u_img, 'Uploaded Image', use_column_width=True)
        # We preprocess the image to fit in algorithm.
        image = np.asarray(u_img)/255
        # print("full image", image)
        image1 = resize(image, (224, 224))
        print("after resizing to (224, 224)", image1.shape)
        image1_expanded = np.expand_dims(image1,axis = 0)
        print("after expanding", image1_expanded.shape)
        
        my_image= resize(image, (64,64)).reshape((1, 64*64*3)).T
        print("shape isss:", my_image.shape)
        print("resizedddd")

    # For newline
    st.write('\n')
        
    if st.button("Click Here to Classify"):
        
        if uploaded_file is None:
            
            st.write("Please upload an Image to Classify")
        
        else:
            
            with st.spinner('Classifying ...'):
                img_array = np.array(u_img)
                img = process_img(u_img)
                print("before predicting shape: ", image1_expanded.shape)
                category_predicted, prediction = predict(image1_expanded)
                first_img_prediction = prediction[0]
                first_img_category = category_predicted[0]
                print("predictedddd")
                time.sleep(2)
                st.success('Done!')
                
            st.header("Algorithm Predicts: ")
            
            #Formatted probability value to 3 decimal places
            probability = "{:.3f}".format(float(first_img_prediction[first_img_category]*100))

            categories = ["c0", "c1", "c2", "c3","c4","c5","c6","c7","c8","c9"]
            categories = ["safe driving", "texting - right",
                        "talking on the phone - right", " texting - left",
                        "talking on the phone - left", "operating the radio", 
                        "drinking", "reaching behind", 
                        "hair and makeup", " talking to passenger"]
            st.write(f"It's a '{categories[first_img_category]}' picture.", '\n' )
            
            st.write('**Probability: **',probability,'%')





