FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-dev \
    libopencv-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install git+https://github.com/facebookresearch/detectron2.git@main
COPY . Detic
WORKDIR Detic
ADD https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
RUN pip3 install -r requirements.txt
# ENTRYPOINT ["python3", "app.py"]
