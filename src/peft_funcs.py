from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoraModel

def use_8b_quantization(model):
    pass

def use_lora(model, config: LoraConfig) -> LoraModel:
    lora_config = LoraConfig(
        **config
    )

    return get_peft_model(model, peft_config=lora_config)