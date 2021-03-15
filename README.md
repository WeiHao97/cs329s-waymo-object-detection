# Detecting Vehicles, Pedestrians and Cyclists
<p float="left">
  <img src="assets/project_logo.png" width="300">
</p>

<H1>Introduction</H1>
<p>
Thanks to the Waymo Open dataset we have created a ML system which,
    detects vehicles, pedestrians and cyclists from driving footage. 
    In this project, we explore the challenges in maintaining and deploying an object detection model. The main goal of this work is to construct a machine learning system that is robust and strives for simplicity and ease of use. We demo the results of our system in the form of a web application that can easilt be deployed on the google cloud platform given the current repository structure.

</p>
  
<p>  
This project is currently under construction 🏗, below you can find the current status of items we are working on:

MVP
- [x] Process and store Waymo data in Coco format on GCP
- [x] Train preliminary models on Waymo data
- [x] Implement Flask application to serve model results
- [x] Create Streamlit application to display results to users
- [x] Deploy web app and model serving application using kubernetes


Demo
- [x] Add exploratory features to the web application
- [x] Enhance the training and evaluation processes (submission of training jobs)
- [ ] Create script to compress and deploy model on Nvidia Jetson Nano
- [ ] Write report
</p>

<H1>Data</H1>
<p>
There are two main sources of data for this project:

- **Waymo Dataset:** https://waymo.com/open/
- **Custom Datasets:** Generated using the Nvidia Jetson Nano
<\p>

<H1>Web Application</H1>
<img src="/assets/webappupdated.jpg" width="800px">
