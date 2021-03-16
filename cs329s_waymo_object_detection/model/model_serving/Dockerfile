FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

# Basic installations
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    git \
    python3.7 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg
RUN apt-get update

RUN echo "deb http://packages.cloud.google.com/apt gcsfuse-bionic main" | tee /etc/apt/sources.list.d/gcsfuse.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y gcsfuse

# Install gcloud.
RUN apt-get install -y apt-transport-https
RUN apt-get install -y ca-certificates
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update
RUN apt-get install -y google-cloud-sdk

RUN mkdir -p /prediction_api
# Copy necessary files to container
# Create folder with prediction api code
COPY . /prediction_api
RUN sudo chmod -R a+rwx /prediction_api

# Install python dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r /prediction_api/requirements.txt

RUN mkdir -p /home/waymo/data
WORKDIR /prediction_api

COPY credentials.json /prediction_api/credentials.json
RUN gcloud auth activate-service-account --key-file='/prediction_api/credentials.json'
RUN gcloud config set project waymo-2d-object-detection
ENV GOOGLE_APPLICATION_CREDENTIALS=/prediction_api/credentials.json
# Start the prediction flask application
ENTRYPOINT ["python3","app.py"]~