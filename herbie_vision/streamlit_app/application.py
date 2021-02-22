import time
import streamlit as st

import numpy as np 
import pandas as pd 
import cv2
    
import PIL.Image as Image
from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt
import streamlit as st


# def vis_image_with_bbox(imgfile):
#     net = model_zoo.get_model('center_net_resnet18_v1b_voc', pretrained=True)
#     #im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
#     #                          'mxnet-ssd/master/data/demo/dog.jpg',
#     #                          path='dog.jpg')
#     x, img = data.transforms.presets.center_net.load_test("tmpImgFile.jpg", short=512)
#     print('Shape of pre-processed image:', x.shape)

#     class_IDs, scores, bounding_boxs = net(x)
#     #fig = plt.figure()
#     ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
#                              class_IDs[0], class_names=net.classes)
#     fig = plt.gcf()
#     #fig.axes.append(ax)
#     plt.draw()
#     fn = 'tmpfile.png'
#     fig.savefig(fn)

#     return fn


def app():
    st.title(":car: Welcome to Herbie Vision :car:")
    img = Image.open('./streamlit_app/assets/herbie.jpg')
    st.image(img, width=600)
    st.markdown('''
        # How does Herbie drive? 
        
        An important aspect of Herbie's driving skills is his visual understanding of objects in his environment. In this app, we are going to give you a sneak peak at some of what Herbie sees.  

        Herbie was willing to provide us with data he uses to identify to the following categories:

        - Other vehicles (cars, buses, trucks etc.)

        - Cyclists

        - Pedestrians 

        There's a lot more  to how Herbie drives but these details are left as an exercise for the user :smiley:.

        ''')
    if st.checkbox('Learn More About Herbie'):
        herbie_url = 'https://www.youtube.com/watch?v=LKyhP7dWGhE'
        st.video(herbie_url)

    st.markdown('''
        # Footage without Herbie Vision
        ''')
    # st.video()
    st.markdown('''
        # Footage with Herbie Vision
        ''')
    # st.video()
    st.markdown('''
        # Your Turn

        Try to upload an image to ask Herbie what he sees:
        ''')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image.save('tmpImgFile.jpg')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        fn = vis_image_with_bbox(image)
        st.image(fn, use_column_width=True)


if __name__ == '__main__':
    app()

   

