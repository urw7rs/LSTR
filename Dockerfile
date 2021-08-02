FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3-opencv

WORKDIR /TuSimple

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/spiralpp

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

COPY environment.txt environment.txt

# Create new environment and install some dependencies.
RUN conda create -y -n lstr --file environment.txt

# Activate environment in .bashrc.
RUN echo "conda activate lstr" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /TuSimple/LSTR
CMD ["/bin/bash", "-c", "python test.py LSTR --testiter 5000 --modality opencv --split testing --batch 1"]
