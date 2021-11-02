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
    def create_mobilenet_s():
        pretrained_model = tensorflow.keras.applications.MobileNetV2(
            input_shape = (224,224,3), 
            include_top = False, #include top gives a classification layer for 1000 classes :O
            weights = 'imagenet', 
            pooling = 'avg' #Ensures that output of pre-trained model is 1-dimensional, output is singlet vector

        )
        pretrained_model.trainable = False #So weights of imagenet will not be changed 
        inputs = pretrained_model.input
        x = layers.BatchNormalization(axis=1)
        x = layers.Dense(128, activation = 'relu')(pretrained_model.output) #128 neurons
        x = layers.Dense(128, activation = 'relu')(x)

        #final layer
        outputs = layers.Dense(10, activation = 'softmax')(x)
        #softmax to make all pr values sum to 1

        model = tensorflow.keras.Model(inputs, outputs)
        return model


    model_mobile = create_mobilenet_s()
    mobile_weights_path = "./saved_weights/mobilenetv2.h5"
    model_mobile.load_weights(mobile_weights_path)
    model_mobile.compile(
        optimizer = 'adam', 
        loss = 'categorical_crossentropy', #data generator has encoded the classes already, so in vector form
        metrics = ['accuracy']
    )
    # print(model_mobile.summary())
    #loading the cat classifier model
    # cat_clf=joblib.load("Cat_Clf_model.pkl")
    cat_clf = model_mobile
    # print("loaded!!")
    #Loading Cat moew sound
    # audio_file = open('Cat-meow.mp3', 'rb')
    # audio_bytes = audio_file.read()

    #functions to predict image
    def sigmoid(z):
        
        s = 1/(1+np.exp(-z))
        
        return s

    from sklearn.metrics import classification_report 
    from sklearn.metrics import confusion_matrix 
    def predict(test_images):
        # w, b, X
        
        # m = X.shape[1]
        # Y_prediction = np.zeros((1,m))
        # w = w.reshape(X.shape[0], 1)
        
        # # Compute the probability of a cat being present in the picture
        
        # Y_prediction = sigmoid((np.dot(w.T, X)+ b))

        print("predictingggg: ")
        predictions = model_mobile.predict(test_images)
        
        Y_prediction = np.argmax(predictions, axis =1 ) #Gets index of class with highest predicted pr

        # #confusion matrix 
        # cm_mobilenet = confusion_matrix(test_images.labels, predictions)
        # lr_mobilenet = classification_report(test_images.labels, predictions, target_names = test_images.class_indices)

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

    # train_images, test_images, val_images = generate_data(tf.keras.applications.mobilenet_v2.preprocess_input)
    # test_images = generate_data(tf.keras.applications.mobilenet_v2.preprocess_input)

    # def process_img(img_path): # here image is file name 
    #     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))

    #     img = tf.keras.preprocessing.image.img_to_array(img)

    #     img = np.expand_dims(img,axis = 0)

    #     return img

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
            
            # Classify cat being present in the picture if prediction > 0.5
            
            # if prediction > 0.5:
            categories = ["c0", "c1", "c2", "c3","c4","c5","c6","c7","c8","c9"]
            categories = ["safe driving", "texting - right",
                        "talking on the phone - right", " texting - left",
                        "talking on the phone - left", "operating the radio", 
                        "drinking", "reaching behind", 
                        "hair and makeup", " talking to passenger"]
            st.write(f"It's a '{categories[first_img_category]}' picture.", '\n' )
            
            st.write('**Probability: **',probability,'%')
                
                # st.sidebar.audio(audio_bytes)
                                
            # else:
            #     st.sidebar.write(" It's a 'Non-Cat' picture ",'\n')
                
            #     st.sidebar.write('**Probability: **',probability,'%')




