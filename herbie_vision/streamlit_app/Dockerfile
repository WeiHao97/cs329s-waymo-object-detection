FROM python:3.7.8-slim

# Expose port which the will be exposed from
EXPOSE 8080
ENV REST_API="http://35.222.90.89/predict"

RUN apt-get update ##[edited]
RUN apt-get install curl ffmpeg libsm6 libxext6 gnupg  -y
RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

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

COPY credentials.json /app/credentials.json
RUN gcloud auth activate-service-account --key-file='/app/credentials.json'
RUN gcloud config set project waymo-2d-object-detection
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json
RUN mkdir /home/data
# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app

# run it!
COPY startup.sh /app/startup.sh
RUN ["chmod", "+x", "/app/startup.sh"]
ENTRYPOINT ["/app/startup.sh"]
