from typing import List, Union
from numpy.typing import NDArray
import numpy as np

from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import SizePartitioner, ShardPartitioner

from parsers import CFLArguments, DatasetArguments, ModelArguments, PEFTArguments


def get_fixed_sample_fn(num_sampled_clients: int = 5, num_samples: int = 10000):
    num_fixed_sample = num_samples / num_sampled_clients
    print(num_fixed_sample)

    def fixed_sample_fn(cids: NDArray[Union[np.int_, np.float_]]):
        return np.full(len(cids), num_fixed_sample)

    return fixed_sample_fn


# preprocess dataset in alpaca format
def alpaca_format_resplitter(dataset: DatasetDict):
    return dataset.map(
        alpaca_format,
        remove_columns=["input", "output"],
        desc=f"Preprocessing dataset for unified format.",
    )


def get_dataloaders(
    cfl_args: CFLArguments,
    dataset_args: DatasetArguments,
    model_args: ModelArguments,
    peft_args: PEFTArguments,
):
    partitioner = ShardPartitioner(
        num_partitions=cfl_args.num_clients,
        partition_by="response",
        num_shards_per_partition=10,
        shard_size=int(dataset_args.dataset_samples
        / (cfl_args.num_clients * cfl_args.fraction_fit)
        / 10),
    )
    fds = FederatedDataset(
        dataset=dataset_args.dataset_name,
        resplitter=alpaca_format_resplitter,
        partitioners={"train": partitioner},
    )

    # partition_sizes = [
    #     len(fds.load_partition(partition_id)) for partition_id in range(cfl_args.num_clients)
    # ]
    # print(partition_sizes)

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    # valloaders = []
    for partition_id in range(cfl_args.num_clients):
        partition = fds.load_partition(partition_id, "train")
        # partition = partition.with_transform(apply_transforms)
        # partition = partition.train_test_split(train_size=0.8, seed=42)
        # trainloaders.append(DataLoader(partition, batch_size=model_args.batch_size))
        trainloaders.append(partition)  # sft trainer accepts dataset instead of dataloader
        # valloaders.append(DataLoader(partition["test"], batch_size=model_args.batch_size))
    # testset = fds.load_split("test").with_transform(apply_transforms)
    # testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders


# may need this
# class CFLDatsetManager


# import pandas as pd

# from conversation import get_conv_template
# from functools import partial

"""to be rewritten"""


def get_dataset(dataset_name, local_data_dir=None):

    if dataset_name in ["gsm8k"]:
        dataset_name = (
            local_data_dir + dataset_name
            if local_data_dir is not None
            else dataset_name
        )
        dataset = load_dataset(dataset_name, split="train", name="main")
    elif dataset_name in ["lighteval/MATH"]:
        dataset_name = (
            local_data_dir + dataset_name
            if local_data_dir is not None
            else dataset_name
        )
        dataset = load_dataset(dataset_name, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = (
            local_data_dir + dataset_name
            if local_data_dir is not None
            else dataset_name
        )
        dataset = load_dataset(dataset_name, split="train_sft")
    else:  # OpenFedLLM - finance
        dataset_name = (
            local_data_dir + dataset_name
            if local_data_dir is not None
            else dataset_name
        )
        dataset = load_dataset(dataset_name, split="train")

    return dataset


def process_sft_dataset(dataset_name: str, dataset: Dataset, dataset_sample: int):
    if dataset_name in [
        "lucasmccabe-lmi/CodeAlpaca-20k",
        "yahma/alpaca-cleaned",
        "FinGPT/fingpt-sentiment-train",
    ]:
        dataset = dataset.map(
            alpaca_format,
            remove_columns=["input", "output"],
            desc=f"Preprocessing {dataset_name} for unified format.",
        )
    # elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
    #     dataset = dataset.rename_column("output", "response")
    # elif dataset_name in [
    #     "tatsu-lab/alpaca",
    #     "vicgalle/alpaca-gpt4",
    #     "gbharti/finance-alpaca",
    # ]:
    #     dataset = dataset.map(
    #         alpaca_format,
    #         remove_columns=["input", "output", "text"],
    #         desc=f"Preprocessing {dataset_name} for unified format.",
    #     )
    # elif dataset_name in ["TIGER-Lab/MathInstruct"]:
    #     df = pd.DataFrame(dataset)
    #     df = df.drop_duplicates(subset=["instruction"])
    #     dataset = datasets.Dataset.from_pandas(df)
    #     dataset = dataset.rename_column("output", "response")
    #     dataset = dataset.remove_columns(["source"])
    # elif dataset_name in ["lighteval/MATH"]:
    #     dataset = dataset.rename_column("solution", "response")
    #     dataset = dataset.rename_column("problem", "instruction")
    #     dataset = dataset.remove_columns(["level", "type"])
    # elif dataset_name in ["gsm8k"]:
    #     dataset = dataset.rename_column("question", "instruction")
    #     dataset = dataset.rename_column("answer", "response")
    # elif dataset_name in [
    #     "medalpaca/medical_meadow_medical_flashcards"
    # ]:  # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
    #     dataset = dataset.remove_columns(["instruction"])
    #     dataset = dataset.rename_column("input", "instruction")
    #     dataset = dataset.rename_column("output", "response")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(
        f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. ====="
    )
    return dataset


def alpaca_format(example):
    if example["input"] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example["input"]
    example["response"] = example["output"]
    return example


# def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
#     if dataset_name in ["Anthropic/hh-rlhf"]:
#         dataset = dataset.map(
#             partial(split_hh, template_name=template_name), load_from_cache_file=False
#         )
#     elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
#         dataset = dataset.map(
#             partial(split_ultrafeedback, template_name=template_name),
#             load_from_cache_file=False,
#         )
#         dataset = dataset.remove_columns(
#             ["prompt_id", "messages", "score_chosen", "score_rejected"]
#         )

#     dataset = dataset.shuffle(seed=2023)
#     if dataset_sample:
#         num_sample = min(len(dataset), dataset_sample)
#         dataset = dataset.select(range(num_sample))
#     print(
#         f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. ====="
#     )
#     print(f">> ===== Data Example =====")
#     print(dataset[0])
#     print(f">> {'='*50}")
#     return dataset


# def find_common_prefix(str1, str2):
#     prefix = ""
#     for i in range(min(len(str1), len(str2))):
#         if str1[i] == str2[i]:
#             prefix += str1[i]
#         else:
#             break
#     return prefix


# def split_ultrafeedback(example, template_name="vicuna_v1.1"):
#     conv_template = get_conv_template(template_name)

#     conv_template.append_message(conv_template.roles[0], example["prompt"])
#     conv_template.append_message(conv_template.roles[1], None)
#     example["prompt"] = conv_template.get_prompt()
#     example["chosen"] = (
#         " " + example["chosen"][1]["content"]
#     )  # There might need a space in the front.
#     example["rejected"] = " " + example["rejected"][1]["content"]
#     return example


# def split_hh(example, template_name="vicuna_v1.1"):
#     common_prefix = find_common_prefix(example["chosen"], example["rejected"])

#     conv_template = get_conv_template(template_name)

#     sentence = common_prefix
#     human_prefix_len = len("\n\nHuman: ")
#     assistant_prefix_len = len("\n\nAssistant: ")
#     sentence = sentence[human_prefix_len:]
#     turn = "user"
#     while True:
#         if turn == "user":
#             index = sentence.find("\n\nAssistant: ")
#             if index == -1:
#                 break
#             else:
#                 conv_template.append_message(conv_template.roles[0], sentence[:index])
#                 turn = "assistant"
#                 sentence = sentence[index + assistant_prefix_len :]
#         elif turn == "assistant":
#             index = sentence.find("\n\nHuman: ")
#             if index == -1:
#                 break
#             else:
#                 conv_template.append_message(conv_template.roles[1], sentence[:index])
#                 turn = "user"
#                 sentence = sentence[index + human_prefix_len :]
#     conv_template.append_message(conv_template.roles[1], None)
#     example["prompt"] = conv_template.get_prompt()
#     example["chosen"] = example["chosen"][
#         len(common_prefix) - 1 :
#     ]  # -1 to include the space in the front.
#     example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
#     return example
