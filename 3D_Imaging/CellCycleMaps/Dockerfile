# Use a base image with Miniconda
FROM continuumio/miniconda3

# get openCV dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && conda clean --all -y

# Set the PATH to include the Conda environment
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# Activate the environment by default
RUN echo "conda activate moffitt" >> ~/.bashrc
ENV PATH=/opt/conda/envs/moffitt/bin:$PATH

WORKDIR /app
COPY . /app

