# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel as base


LABEL author="Cheng Liang" \
      flower.simulation="true" \
      description="test evironment for Flower framework with one computing machine"

# Install dependencies
RUN apt update && apt install build-essential
RUN pip install flwr[simulation] flwr_datasets[vision]
RUN pip install -U huggingface_hub[cli]
RUN pip install transformers trl sentencepiece
RUN pip install peft bitsandbytes
RUN pip install accelerate
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

# huggingface login stage
# make sure to store your token at ./hf_token
FROM base as login-stage
COPY .hf_token /root/.hf_token
RUN huggingface-cli login --token $(cat /root/.hf_token)

FROM base
COPY --from=login-stage \
     /root/.cache/huggingface \
     /root/.cache/huggingface