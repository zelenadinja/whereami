#! /bin/bash

sudo apt-get update
sudo apt install python3-pip
pip3 install kaggle
scp -i ec2keypair.pem Downloads/kaggle.json ubuntu@ec2-3-70-209-54.eu-central-1.compute.amazonaws.com:~/.kaggle/
chmod 600 /home/ubuntu/.kaggle/kaggle.json
alias kaggle=/home/ubuntu/.kaggle/kaggle.json
kaggle competitions download -c landmark-recognition-2021
sudo apt install unzip
unzip ladnmark-recognition-2021.zip
aws s3api get-object --bucket landmarkdataset --key artifacts/not_used_object_keys.txt not_used_object_keys.txt
time xargs -a not_used_object_keys.txt rm
aws s3 cp train/ s3://landmarkdataset/train --recursive
