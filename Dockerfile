FROM nvidia/cuda:9.2-runtime-ubuntu18.04
FROM pytorch/pytorch

# Install some basic utilities 
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    ffmpeg \
    vim \
    && rm -rf /var/lib/apt/lists/*


# Create a working directory 
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it 
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory 
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda 
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH 
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment 
RUN /home/user/miniconda/bin/conda install conda-build \
    && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.6 \
    && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36 
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV 
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Upgrade pip
RUN pip install --upgrade pip

#Install Requests, a Python library for making HTTP requests
RUN conda install -y requests && \
    conda clean -ya

# Copying files
ADD rirnet /app/rirnet/
ADD audio /app/audio/
RUN mkdir /app/database/
ADD database/_default.yaml /app/database/_default.yaml
ADD setup.py /app/setup.py

# Install python packages
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install pyroomacoustics
RUN pip install .
