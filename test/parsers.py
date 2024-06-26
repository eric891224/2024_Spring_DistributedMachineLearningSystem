from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

# parser = ArgumentParser()

# parser.add_argument('--num_clients', default=50, type=int)
# parser.add_argument("--algorithm", "-a", default="fedavg", type=str)
# parser.add_argument('--num_global_rounds', default=200, type=int)
# parser.add_argument('--num_local_epochs', default=10, type=int)

# parser.add_argument('--dataset_name', default='FinGPT/fingpt-sentiment-train', type=str)
# parser.add_argument('--dataset_samples', default=10000, type=int)
# parser.add_argument('--model_name', default="meta-llama/Llama-2-7b-hf", type=str)


@dataclass
class CFLArguments:
    algorithm: Optional[str] = field(
        default="fedavg", metadata={"help": "FL algorithm"}
    )
    num_clients: Optional[int] = field(
        default=30, metadata={"help": "number of clients"}
    )
    fraction_fit: Optional[float] = field(
        default=1 / 6, metadata={"help": "fraction of clients to be sampled"}
    )
    num_global_rounds: Optional[int] = field(
        default=200, metadata={"help": "number of FL rounds"}
    )
    num_local_epochs: Optional[int] = field(
        default=10, metadata={"help": "number of client epochs"}
    )


@dataclass
class DatasetArguments:
    dataset_name: Optional[str] = field(
        default="FinGPT/fingpt-sentiment-train", metadata={"help": "HF dataset name"}
    )
    dataset_samples: Optional[int] = field(
        default=10000, metadata={"help": "number of samples to use from the dataset"}
    )


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "HF model name"}
    )
    learning_rate: Optional[float] = field(
        default=2e-5, metadata={"help": "the learning rate"}
    )  # vicuna and alpaca use 2e-5
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    max_seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    template: Optional[str] = field(
        default="alpaca", metadata={"help": "the template to use"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    save_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )


@dataclass
class PEFTArguments:
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )

    use_lora: Optional[bool] = field(default=False, metadata={"help": "use LoRA"})
    lora_r: Optional[int] = field(
        default=8, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the dropout parameter of the LoRA adapters"}
    )


cfl_parser = HfArgumentParser(
    (CFLArguments, DatasetArguments, ModelArguments, PEFTArguments)
)
