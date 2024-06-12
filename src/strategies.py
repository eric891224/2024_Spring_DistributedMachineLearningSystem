from flwr.server import strategy

from parsers import CFLArguments, DatasetArguments, ModelArguments, PEFTArguments


def get_strategy(
    cfl_args: CFLArguments,
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    peft_args: PEFTArguments,
) -> strategy.Strategy:
    # initial_parameters
    if cfl_args.algorithm == "fedavg":
        return strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0,
            min_fit_clients=5,
            min_evaluate_clients=0,
            min_available_clients=cfl_args.num_clients,
            # initial_parameters=
        )
