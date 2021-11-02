import streamlit as st
st.set_page_config(layout='wide')
from multiapp import MultiApp
import app1_introduction
import app2_face_blur  
import app3_class_activation_maps
import app4_predict_mobilenet
import app5_predict_resnet
# , methodology_app, prototype_app

app = MultiApp()

# Navigation
app.add_app("Actual Idiots", app1_introduction.app)
app.add_app("Face Blurring", app2_face_blur.app)
app.add_app("Class Activation Maps", app3_class_activation_maps.app)
app.add_app("Model Prediction", app4_predict_mobilenet.app)
# app.add_app("Model Prediction", app5_predict_resnet.app)
# app.add_app("Methodology", methodology_app.app)
# app.add_app("Prototype", prototype_app.app)

# Main App
app.run()
