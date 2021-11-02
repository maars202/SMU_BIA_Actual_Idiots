import streamlit as st

from multiapp import MultiApp
import home_app_1
import face_blur_app_2  # , methodology_app, prototype_app
import mobile_prediction
app = MultiApp()

# Navigation
app.add_app("Actual Idiots", home_app_1.app)
app.add_app("Face Blurring", face_blur_app_2.app)
app.add_app("Model Prediction", mobile_prediction.app)
# app.add_app("Methodology", methodology_app.app)
# app.add_app("Prototype", prototype_app.app)
st.set_page_config(layout='wide')
# Main App
app.run()
