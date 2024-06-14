from typing import Tuple
from logging import INFO, DEBUG

import torch
import flwr as fl
from flwr.client import ClientFn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training

from config import DEVICE, client_resources
from parsers import (
    cfl_parser,
    CFLArguments,
    DatasetArguments,
    ModelArguments,
    PEFTArguments,
)
from virtual_client_engine import get_client_fn
from dataset import get_dataloaders
from peft_funcs import get_automodel_config, apply_lora
from strategies import get_strategy

from client import get_peft_parameters, set_peft_parameters, CFLClient
from trainer import get_sft_config
from flwr.common import FitIns
from trainer import get_data_collator
from template import get_formatting_prompts_func
import copy
from flwr.common.logger import log

print(
    f"Running on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

fl.common.logger.configure(identifier="CFL", filename="log.txt")

if __name__ == "__main__":
    args: Tuple[CFLArguments, DatasetArguments, ModelArguments, PEFTArguments] = (
        cfl_parser.parse_args_into_dataclasses()
    )
    cfl_args, dataset_args, model_args, peft_args = args

    # load dataset
    trainloaders = get_dataloaders(*args)
    # print(
    #     f"{len(trainloaders)} trainloaders with each containing {len(trainloaders[0].dataset)} samples, batch size {len(trainloaders[0])}"
    # )
    print(
        f"{len(trainloaders)} datasets divisions with each containing {len(trainloaders[0])} samples"
    )

    # get model configs
    device_map, quantization_config, torch_dtype = get_automodel_config(peft_args)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )

    # apply quantization
    if peft_args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # apply LoRA
    if peft_args.use_lora:
        model = apply_lora(model, peft_args)
        model.print_trainable_parameters()
    print(model)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name, use_fast=False, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  # following vicuna

    # flower virtual client engine for simulation
    # client_fn: ClientFn = get_client_fn(model, tokenizer, trainloaders, *args)
    formatting_func, response_template = get_formatting_prompts_func(
        model_args.template, tokenizer.eos_token
    )
    data_collator = get_data_collator(
        tokenizer=tokenizer, response_template=response_template
    )

    def client_fn(cid: str) -> CFLClient:
        """Create a Flower client representing a single organization."""
        # Load Model
        log(INFO, f"Clint {cid} test")
        m = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        m = prepare_model_for_kbit_training(m)
        m = apply_lora(m, peft_args)
        m.print_trainable_parameters()

        m.config.use_cache = False
        # Load data
        trainloader = trainloaders[int(cid)]

        # Create a single Flower client representing a single organization
        return CFLClient(
            cid=cid,
            model=m,
            tokenizer=tokenizer,
            trainloader=trainloader,
            valloader=None,
            use_lora=peft_args.use_lora,
            formatting_func=formatting_func,
            data_collator=data_collator,
        )

    # get flower strategy
    strategy = get_strategy(model, *args)

    # start flower simulation
    # sft_config = get_sft_config(cfl_args=cfl_args, model_args=model_args)
    # config = {
    #     "server_round": 1,
    #     "local_epochs": cfl_args.num_local_epochs,
    #     "sft_config": sft_config,
    # }
    # client_fn('0').get_parameters({})
    # client_fn("0").fit(FitIns(parameters=get_peft_parameters(model), config=config))
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfl_args.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfl_args.num_global_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
