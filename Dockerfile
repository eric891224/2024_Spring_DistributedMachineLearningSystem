# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel


LABEL author="Cheng Liang" \
      flower.simulation="true" \
      description="test evironment for Flower framework with one computing machine"


RUN apt update && apt install build-essential
RUN pip install flwr[simulation] flwr_datasets[vision]
RUN pip install transformers trl
RUN pip install peft