import copy
from typing import List

from torch import nn
from torch.utils.data import DataLoader
from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from flwr.client import ClientFn
from logging import INFO, DEBUG
from flwr.common.logger import log
from transformers import AutoModelForCausalLM, PreTrainedTokenizer

from parsers import CFLArguments, DatasetArguments, ModelArguments, PEFTArguments
from client import CFLClient
from trainer import get_data_collator
from template import get_formatting_prompts_func

from models import Net

from peft_funcs import get_automodel_config, apply_lora
from peft import prepare_model_for_kbit_training


def get_client_fn(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    trainloaders: List[DataLoader],
    cfl_args: CFLArguments,
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    peft_args: PEFTArguments,
) -> ClientFn:
    # formatting function for TRL SFTTrainer
    formatting_func, response_template = get_formatting_prompts_func(
        model_args.template, tokenizer.eos_token
    )
    data_collator = get_data_collator(
        tokenizer=tokenizer, response_template=response_template
    )

    def client_fn(cid: str) -> CFLClient:
        """Create a Flower client representing a single organization."""
        # Load Model
        device_map, quantization_config, torch_dtype = get_automodel_config(peft_args)
        # model = AutoModelForCausalLM.from_pretrained(
        #     pretrained_model_name_or_path=model_args.model_name,
        #     quantization_config=quantization_config,
        #     device_map=device_map,
        #     torch_dtype=torch_dtype,
        # )
        # model = prepare_model_for_kbit_training(model)
        # model = apply_lora(model, peft_args)
        # model.print_trainable_parameters()
        # model.config.use_cache = False
        # model = copy.deepcopy(model)

        # Load data
        trainloader = trainloaders[int(cid)]

        # Create a single Flower client representing a single organization
        return CFLClient(
            cid=cid,
            model=Net(),
            tokenizer=tokenizer,
            trainloader=trainloader,
            valloader=None,
            use_lora=peft_args.use_lora,
            formatting_func=formatting_func,
            data_collator=data_collator,
        ).to_client()

    return client_fn
