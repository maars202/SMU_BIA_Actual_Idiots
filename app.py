import streamlit as st
st.set_page_config(layout='wide')
from multiapp import MultiApp
import app1_introduction
import app2_face_blur  
import app3_class_activation_maps
import app4_predict_mobilenet
import app5_predict_resnet
import app6_predict_densenet
# , methodology_app, prototype_app

app = MultiApp()

# Navigation
app.add_app("Actual Idiots", app1_introduction.app)
app.add_app("Face Blurring", app2_face_blur.app)
app.add_app("Class Activation Maps", app3_class_activation_maps.app)
# st.sidebar()
app.add_app("Model Prediction Mobilenet", app4_predict_mobilenet.app)
# app.add_app("Model Prediction Resnet", app5_predict_resnet.app)
# app.add_app("Model Prediction Densenet", app6_predict_densenet.app)
# app.add_app("Methodology", methodology_app.app)
# app.add_app("Prototype", prototype_app.app)

# Main App
app.run()

