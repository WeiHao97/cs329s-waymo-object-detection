import os
import time
import json
import requests
import streamlit as st

import numpy as np 
import pandas as pd 
import cv2    
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from herbie_vision.utils.image import plot_annotations


def generate_prediction_image(imgfile, rest_api):
    """
    This function requests prediction from rest_api and 
    plots bounding boxes of result over image.
    """
    save_path = 'prediction.jpeg'

    # Make request to prediction api
    files = {'image': open(imgfile, 'rb')}
    response = requests.post(rest_api, files=files)

    # Parse response
    annotations = json.loads(response.content)
    boxes = [np.array(x).astype(float) for x in annotations['boxes']]
    labels = np.array(annotations['labels']).astype('int')
    scores = np.array(annotations['scores']).astype('float')

    # Create new figure
    img = Image.open(imgfile)
    pred_fig = plot_annotations(img, boxes, labels, scores, 
                                0.6, save_fig_path=save_path)

    return save_path


def app():
    # URLs used for video footage
    herbie_url = 'https://www.youtube.com/watch?v=LKyhP7dWGhE'
    original_footage = 'https://www.youtube.com/watch?v=VjDbvm6Nd7Y'
    updated_footage = 'https://www.youtube.com/watch?v=wyrAa57rOWs'
    rest_api = os.environ['REST_API']
    
    # App header
    st.title("Welcome to Herbie Vision")
    img = Image.open('./assets/herbie.jpg')
    st.image(img, width=600)

    # Section 1
    st.markdown('''# How does Herbie drive? ''')
    if st.checkbox('Learn more about Herbie'):
        st.markdown('''
            An important aspect of Herbie's driving skills is his visual understanding of objects in his environment. In this app, we are going to give you a sneak peak at some of what Herbie sees.  

        Herbie was willing to provide us with access to the model he uses to identify to the following objects:

        - Vehicles (cars, buses, trucks etc.)

        - Cyclists

        - Pedestrians 

        There's a lot more objects which Herbie could identify as well as a lot more details to how Herbie drives but these are left as an exercise for the user :smiley:.
            ''')
        st.video(herbie_url)

    # Section 2
    st.markdown('''# What Herbie sees...''')
    if st.checkbox('Take a look'):
        st.markdown('''
            ## Before
            ''')
        st.video(original_footage)
        st.markdown('''
            ## After
            ''')
        st.video(updated_footage)

    # Section 3
    st.markdown('''
        # Try with your own Image

        Try to upload an image to ask Herbie what he sees:
        ''')

    if st.checkbox('Give it a go'):
        uploaded_file = st.file_uploader("Choose an image...", type="jpeg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image.save('tmpImgFile.jpg')
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Asking Herbie for his opinion...")
            pred_img = generate_prediction_image('tmpImgFile.jpg', rest_api)
            st.image(pred_img, caption="What herbie sees in the image",use_column_width=True)


if __name__ == '__main__':
    app()

   

