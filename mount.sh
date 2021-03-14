#! /bin/bash
sudo apt-get -y update
sudo apt-get -y install nfs-common git
sudo mkdir -p /home/waymo/data
sudo gcsfuse --implicit-dirs waymo-processed /home/waymo/data
# sudo mount 10.43.112.242:/datasets /home/waymo/data
sudo chmod 777 /home/waymo/data
sudo apt-get -y install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip python-setuptools
