import math
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from peft.utils import prepare_model_for_kbit_training

from peft_funcs import get_automodel_config

from parsers import CFLArguments, DatasetArguments, ModelArguments, PEFTArguments
import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(
    cfl_args: CFLArguments,
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    peft_args: PEFTArguments,
):
    """Load model with appropriate quantization config and other optimizations.

    Please refer to this example for `peft + BitsAndBytes`:
    https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py
    """

    # if model_cfg.quantization == 4:
    #     quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    # elif model_cfg.quantization == 8:
    #     quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # else:
    #     raise ValueError(
    #         f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
    #     )

    # get model configs
    device_map, quantization_config, torch_dtype = get_automodel_config(peft_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )

    # apply quantization
    if peft_args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_args.gradient_checkpointing
        )

    # apply LoRA
    if peft_args.use_lora:
        peft_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            task_type="CAUSAL_LM",
            bias="none",
        )
        return get_peft_model(model, peft_config)

    return model
