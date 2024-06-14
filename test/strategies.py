from torch import nn
from flwr.common import ndarrays_to_parameters
from flwr.server import strategy

from parsers import CFLArguments, DatasetArguments, ModelArguments, PEFTArguments
from client import get_peft_parameters, get_parameters
from trainer import get_sft_config


def get_strategy(
    model: nn.Module,
    cfl_args: CFLArguments,
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    peft_args: PEFTArguments,
) -> strategy.Strategy:
    if peft_args.use_lora:
        initial_parameters = ndarrays_to_parameters(get_peft_parameters(model))
    else:
        initial_parameters = ndarrays_to_parameters(get_parameters(model))

    sft_config = get_sft_config(cfl_args=cfl_args, model_args=model_args)

    def on_fit_config_fn(server_round: int):
        sft_config.learning_rate = cosine_learning_rate(
            server_round=server_round, total_rounds=cfl_args.num_global_rounds
        )
        config = {
            "server_round": server_round,
            "local_epochs": cfl_args.num_local_epochs,
            'sft_config': sft_config
        }

        return config

    if cfl_args.algorithm == "fedavg":
        return strategy.FedAvg(
            fraction_fit=cfl_args.fraction_fit,
            fraction_evaluate=0,
            min_fit_clients=5,
            min_evaluate_clients=0,
            min_available_clients=cfl_args.num_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=on_fit_config_fn,
        )


import math


def cosine_learning_rate(server_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    :param current_round: The current training round (0-indexed).
    :param total_rounds: The total number of training rounds.
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :return: The computed learning rate for the current round.
    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (
        1 + math.cos(math.pi * server_round / total_rounds)
    )
    return cosine_lr