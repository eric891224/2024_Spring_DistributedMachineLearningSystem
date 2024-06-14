import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
else:
    DEVICE = torch.device("cpu")
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}