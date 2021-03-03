# Herbie Vision
<p float="left">
  <img src="assets/herbie.jpg" width="300">
</p>

<H1>Introduction</H1>
<p>
In this project, we explore the challenges in maintaining and deploying an object detection model. The main goal of this work is to construct a machine learning system that is robust and strives for simplicity and ease of use. We demo the results of our system in the form of a web application that can easilt be deployed on the google cloud platform given the current repository structure.
<\p>
  
<p>  
This project is currently under construction 🏗, below you can find the current status:

- [x] Process and store Waymo data in GCP
- [x] Train preliminary models on Waymo data
- [x] Implement Flask application to serve model results
- [x] Create Streamlit application to display results to users
- [x] Kubernetes config + Makefile
- [ ] Include github actions for redeployments
- [ ] Include Dataturks annotation app 
- [ ] Build + integrate model training setup with Weights and Biases
- [ ] Setup monitoring and logging
- [ ] Create script for model compression to deploy on Nvidia Jetson Nano
<\p>

<H1>Data</H1>
There are two main sources of data for this project:
**Waymo Dataset:** https://waymo.com/open/
**Customer Datasets:** Generated using the Nvidia Jetson Nano

<H1>Web Application</H1>
![](https://github.com/peterdavidfagan/herbie-vision/blob/main/assets/webapp.gif)
