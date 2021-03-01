FROM ubuntu:latest

# Set timezone
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

# Copy necessary files to container
RUN mkdir -p /prediction_api
COPY requirements.txt /prediction_api
COPY creds.json /prediction_api

# Install python dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r /prediction_api/requirements.txt

# Create folder with prediction api code
COPY . /prediction_api
RUN sudo chmod -R a+rwx /prediction_api

WORKDIR /prediction_api

# Start the prediction flask application
CMD ["python3","app.py"]~