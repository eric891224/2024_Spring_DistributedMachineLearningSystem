from typing import List, Callable
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from trl import SFTTrainer
from transformers import PreTrainedTokenizer
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from flwr.client import Client
from flwr.common import (
    Status,
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
)
from logging import INFO, DEBUG
from flwr.common.logger import log

from config import DEVICE


class CFLClient(Client):
    def __init__(
        self,
        cid: int,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        trainloader: DataLoader,
        valloader: DataLoader = None,
        use_lora: bool = False,
        formatting_func: Callable = None,
        data_collator=None,
    ):
        self.cid = cid
        self.model = model
        self.tokenizer = tokenizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.use_lora = use_lora
        self.formatting_func = formatting_func
        self.data_collator = data_collator

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")

        # Serialization
        if self.use_lora:
            parameters = get_peft_parameters(self.model)
        else:
            parameters = get_parameters(self.model)

        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(status=status, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, f"[Client {self.cid}] fit, config: {ins.config}")

        # Deserialization and update model weight
        if self.use_lora:
            set_peft_parameters(self.model, ins.parameters)
        else:
            set_parameters(self.model, ins.parameters)

        # train
        # trainer = SFTTrainer(
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        #     args=ins.config["sft_config"],
        #     train_dataset=self.trainloader,
        #     formatting_func=self.formatting_func,
        #     data_collator=self.data_collator,
        # )
        # result = trainer.train()
        result = "None"
        log(
            INFO,
            f"[Client {self.cid}] fit, round: {ins.config['server_round']}, result: {result}, config: {ins.config}",
        )

        # Serialization
        if self.use_lora:
            updated_parameters = get_peft_parameters(self.model)
        else:
            updated_parameters = get_parameters(self.model)

        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=updated_parameters,
            num_examples=len(self.trainloader),
            metrics={},  # TODO
        )

    # TODO
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # return super().evaluate(ins)
        status = Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="Evaluate TODO")
        return EvaluateRes(status=status, loss=0, num_examples=0)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def get_peft_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in get_peft_model_state_dict(model).items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def set_peft_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(get_peft_model_state_dict(model).keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)
