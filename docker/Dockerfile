FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN apt update

RUN apt install -y g++ \
         make \
         wget \
         unzip \
         vim \
         git \
         htop \
         dstat \
         python3-pip

RUN wget \
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   && bash Miniconda3-latest-Linux-x86_64.sh -b -p /conda \
   && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=${PATH}:/conda/bin

RUN mkdir /working_dir
WORKDIR /working_dir

ENV TOKENIZERS_PARALLELISM=true

COPY environment.yml .

RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "unsupervised_kg", "/bin/bash", "-c"]

# COPY ./ ./
