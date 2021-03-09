FROM tensorflow/tensorflow:latest

ENV DEBIAN_FRONTEND=noninteractive

# Basic installations python3.6 \  python3-pip \
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    ffmpeg
RUN apt-get update

RUN mkdir -p /home/waymo
RUN mkdir -p /home/waymo/temp
# Copy necessary files to container
COPY requirements.txt /home/waymo
COPY credentials.yaml /home/waymo
COPY data_processing.yaml /home/waymo
COPY credentials.json /home/waymo

# Install python dependencies
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --upgrade pip
RUN pip3 install -r /home/waymo/requirements.txt

# Create folder with prediction api code
COPY . /home/waymo
RUN sudo chmod -R a+rwx /home/waymo

WORKDIR /home/waymo

# Start the prediction flask application
ENTRYPOINT ["python","waymo_to_coco.py","/home/waymo/credentials.yaml","/home/waymo/data_processing.yaml"]