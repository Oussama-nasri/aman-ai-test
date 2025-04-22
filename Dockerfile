# Use a base image with Miniconda
FROM continuumio/miniconda3:4.9.2

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Update conda and create a new environment with Python 3.6
RUN conda update -n base conda && \
    conda create -y --name aman python=3.6

# Initialize Conda for the shell
RUN conda init bash

# Configure Conda channels explicitly
RUN conda config --add channels defaults && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict

# Activate the environment and install rdkit with a specific compatible version
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate aman && \
    conda install -c conda-forge rdkit=2021.09.5

# Install the project dependencies from setup.py
RUN conda activate aman && \
    pip install .

# Expose the port for the Flask app
EXPOSE 5000

# Set the command to run the Flask app
CMD ["conda", "run", "--no-capture-output", "-n", "aman", "python", "app.py"]