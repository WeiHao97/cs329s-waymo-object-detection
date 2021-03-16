import streamlit as st

from PIL import Image
from cs329s_waymo_object_detection.utils.image import plot_annotations

import requests
import json
import numpy as np
from glob import glob


def generate_prediction_user_image(imgfile, rest_api):
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


def generate_prediction_all_cameras(imgfiles, rest_api):
    """
    This function requests prediction from rest_api and 
    plots bounding boxes of result over image.
    """
    save_path = 'prediction.jpeg'

    # Make request to prediction api
    files = {'images': [open(imgfile, 'rb') for image in imgfiles]}
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
################################################################################

st.set_page_config(page_title="Awesome Object Detection",  
                    page_icon="car",
                    layout="wide")

# HEADER SECTION
header_img = Image.open('./assets/project_logo.png')
st.image(header_img, width=150)
row1_1, row1_2 = st.beta_columns((2,3))    

with row1_1:
    st.title("Detecting Vehicles, Pedestrians and Cyclists")

with row1_2:
    st.write(
    """
    ##
    Thanks to the Waymo Open dataset we have created a ML system which,
    detects vehicles, pedestrians and cyclists from driving footage. In this app,
    we demonstrate the current capabilities of our system. 
    For more details on the data used for this project please consult: https://waymo.com/open/.
    """)

# DROP DOWN SELECTS
row2_1, row2_2, row2_3 = st.beta_columns((1.5,1.5,1.5))
location_map = {'San Francisco':'location_sf','Phoenix':'location_phx','Other':'location_other'} 
tod_map = {'Day':'day','Dawn/Dusk':'dawn_dusk','Night':'night'} 
weather_map = {'Sunny':'sunny', 'Rain':'rain'}
with row2_1:
    location = st.selectbox('Location', ('San Francisco', 'Phoenix', 'Other'))
with row2_2:
    tod = st.selectbox('Time of Day', ('Day', 'Dawn/Dusk', 'Night'))
with row2_3:
    weather = st.selectbox('Weather', ('Sunny', 'Rain'))

# Frame Slider
frame = st.slider("Frame",0,180,0)
st.markdown("#")
try:
    segments = list(glob('/home/waymo/data/{}/{}/{}/*/'.format(location_map[location],tod_map[tod], weather_map[weather])))
    segment = segments[np.random.randint(0,len(segments))]
except:
    segment = None
#Plotting Driving Segment With Predictions
row3_1, row3_2, row3_3 = st.beta_columns((1.5,1.5,1.5))
with row3_1:
    try:
        img_fl = Image.open('/home/waymo/data/{}/{}/{}/{}_{}_FRONT_LEFT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment,frame))
        st.markdown("## Front Left Camera")
        st.image(img_fl)
    except:
        st.markdown("")


with row3_2:
    try:
        img_f = Image.open('/home/waymo/data/{}/{}/{}/{}_{}_FRONT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment,frame))
        st.markdown("## Front Center Camera")
        st.image(img_f)
    except:
        st.markdown("")
with row3_3:
    try:
        img_fr = Image.open('/home/waymo/data/{}/{}/{}/{}_{}_FRONT_RIGHT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment,frame))
        st.markdown("## Front Right Camera")
        st.image(img_fr)
    except:
        st.markdown("")

row4_1, row4_2, row4_3 = st.beta_columns((1.5, 1.5,1.5))
with row4_1:
    try:
        img_l = Image.open('/home/waymo/data/{}/{}/{}/{}_{}_SIDE_LEFT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment,frame))
        st.markdown("## Left Camera")
        st.image(img_l)
    except:
        st.markdown("")

with row4_2:
    st.markdown("")
    
with row4_3:
    try:
        img_r = Image.open('/home/waymo/data/{}/{}/{}/{}_{}_SIDE_RIGHT.jpeg'.format(location_map[location],tod_map[tod], weather_map[weather], segment,frame))
        st.markdown("## Right Camera")
        st.image(img_r)
    except:
        st.markdown("")


st.markdown("#")
# USER UPLOAD SECTION
st.markdown('''# Your Turn''')
st.selectbox('Model Selection', ('Base', 'Nighttime', 'Rain'))

st.markdown('''## Try your own image''')
uploaded_img = st.file_uploader("Upload an image...", type="jpeg")
if uploaded_img is not None:
    image = Image.open(uploaded_img)
    image.save('tmpImgFile.jpg')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Our Model is running its prediction...")
    pred_img = generate_prediction_user_image('tmpImgFile.jpg', rest_api)
    st.image(pred_img, caption="The bouding box predictions",use_column_width=True)


st.markdown('#')
# MODEL EVALUTION SECTION
st.markdown('''# Model Evaluation''')
st.markdown('#')
st.markdown('We are big fans of Weights and Biases, you may find some further details on the training runs we ran below:')
st.image('./assets/wandb.png',width=400)
st.components.v1.iframe('https://wandb.ai/peterdavidfagan/waymo-2d-object-detection?workspace=user-peterdavidfagan',height=900, scrolling = True)

st.markdown('#')
# MODEL DEPLOYMENT SECTION
st.markdown('''# Model Deployment''')
st.markdown('Nvidia Jetson Nano Youtube Video ?')

st.markdown('#')
# TEAM/ABOUT SECTION
st.markdown('''# The Team''')

row5_1, row5_2, row5_3 = st.beta_columns((1.5,1.5,1.5))

with row5_1:
    st.write("""
        ## Ethan
        Interested in computer vision, aerospace, and human-computer interaction. Passionate about leveraging these 
        to create a cleaner, smarter, and healthier world. Outside of the classroom, I spend my free time learning new songs on guitar,
        walking dogs, snowboarding, playing basketball, backpacking, watching The Office, and hanging out with friends + family.
        """)
with row5_2:
    st.write("""
        ## Peter
        Interested in robotics and how we can leverage reinforcement learning to tackle challenging robot control problems.
        Passionate about realizing the positive impacts artificial intelligence can have on society. Outside of these topics
        I like to bike, swim and lift weights. 
        """)
with row5_3:
    st.write("""
        ## Tiffany
        Passionate about leveraging computer science technology for social impact and technical exploration. 
        Interested in dogs, music composition, (easy) hikes, and how to balance software engineering and alpaca farming 
        careers long term. 
        """)
