# Warning: Docker file will not work as PAT required has been removed.
# File included only for reference.
# Description: This file is used to create a Docker image for the project.

# request a platform

# Use the official pytorch image.
# https://hub.docker.com/_/python
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git

ARG GIT_USERNAME="michaelbennett99"
ARG GIT_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ARG GIT_REPO="github.com/michaelbennett99/CPSC452_project.git"

# Clone the project repository.
RUN git clone https://$GIT_USERNAME:$GIT_TOKEN@$GIT_REPO .

# create update pip
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Update some cv libraries
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
