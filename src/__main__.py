import torch
import flwr as fl

from config import DEVICE, client_resources
from parsers import cfl_parser
from virtual_client_engine import get_client_fn
from dataset import get_dataloaders
from strategies import get_strategy


print(
    f"Running on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

if __name__ == "__main__":
    args = cfl_parser.parse_args_into_dataclasses()
    cfl_args, dataset_args, model_args, peft_args = args

    trainloaders, valloaders = get_dataloaders(**args)
    strategy = get_strategy(**args)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(),
        num_clients=cfl_args.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfl_args.num_global_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
