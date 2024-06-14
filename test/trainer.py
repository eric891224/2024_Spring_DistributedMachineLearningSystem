from trl import SFTConfig, DataCollatorForCompletionOnlyLM

from parsers import CFLArguments, DatasetArguments, ModelArguments, PEFTArguments


def get_sft_config(
    cfl_args: CFLArguments,
    model_args: ModelArguments,
):
    return SFTConfig(
        output_dir=model_args.output_dir,
        per_device_train_batch_size=model_args.batch_size,
        gradient_accumulation_steps=model_args.gradient_accumulation_steps,
        num_train_epochs=cfl_args.num_local_epochs,
        save_steps=model_args.save_steps,
        save_total_limit=model_args.save_total_limit,
        lr_scheduler_type="constant",
        max_seq_length=model_args.max_seq_length
    )


def get_data_collator(tokenizer, response_template):
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]
    # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]` for Llama2
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return data_collator


# def get_training_args(script_args, new_lr):
#     training_args = TrainingArguments(
#         output_dir=script_args.output_dir,
#         per_device_train_batch_size=script_args.batch_size,
#         gradient_accumulation_steps=script_args.gradient_accumulation_steps,
#         learning_rate=new_lr,
#         logging_steps=script_args.logging_steps,
#         num_train_epochs=script_args.num_train_epochs,
#         max_steps=script_args.max_steps,
#         report_to=script_args.log_with,
#         save_steps=script_args.save_steps,
#         save_total_limit=script_args.save_total_limit,
#         push_to_hub=script_args.push_to_hub,
#         hub_model_id=script_args.hub_model_id,
#         gradient_checkpointing=script_args.gradient_checkpointing,
#         lr_scheduler_type="constant",
#     )
#     return training_args
