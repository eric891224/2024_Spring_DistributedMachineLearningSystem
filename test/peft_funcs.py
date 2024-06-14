import torch
from accelerate import Accelerator
from transformers import BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    get_peft_model_state_dict
)

from parsers import PEFTArguments


def get_automodel_config(peft_args: PEFTArguments):
    device_map, quantization_config, torch_dtype = None, None, None

    if peft_args.load_in_8bit:
        device_map = {"": Accelerator().local_process_index}
        quantization_config = BitsAndBytesConfig(load_in_8bit=peft_args.load_in_8bit)
        torch_dtype = torch.bfloat16

    return device_map, quantization_config, torch_dtype


def apply_lora(model: torch.nn.Module, peft_args: PEFTArguments):
    print('applying LoRA...')
    return get_peft_model(model, get_lora_config(peft_args))


def get_lora_config(peft_args: PEFTArguments):
    return LoraConfig(
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


# def use_lora(model, config: LoraConfig) -> LoraModel:
#     lora_config = LoraConfig(**config)

#     return get_peft_model(model, peft_config=lora_config)
