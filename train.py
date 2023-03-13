import os
import wandb
import argparse

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

config = {
    
    "dataset_id":"samsum",
    "model_id": "google/flan-t5-base",

    # Trainer wandb environment variables
    "wandb_log_model": "checkpoint",
    "wandb_watch" : 'false',

    # Trainer arguments
    "learning_rate": 5e-5,
    "per_device_train_batch_size":8,
    "per_device_eval_batch_size":8,
    "gradient_accumulation_steps":1,
    "max_steps":1000,
    "warmup_steps":100,
    "num_train_epochs":1,
    "logging_steps":5,
    "fp16":False,
    "run_name" : None,
    "evaluation_strategy":"steps", # "epoch",
    "eval_steps":100,
    "save_total_limit":2,
    "save_strategy": "steps", # "epoch",
   
    # Debugging
    "debug_mode" : True,
    "debug_dataset_indices" : list(range(128))

}

config["output_dir"] = f"{config['model_id'].split('/')[1]}-{config['dataset_id']}"

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default='launch-demo-flan-t5',
        required=True,
        help="Weights & Biases Project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        required=False,
        help="Weights & Biases Team or Username to log to",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    # START A WANDB RUN
    run = wandb.init(entity=args.wandb_entity,
            project=args.wandb_project,
            config=config)
    run.log_code()

    # Set environment variables for HF Trainer's wandb logging
    os.environ["WANDB_LOG_MODEL"] = wandb.config.wandb_log_model
    os.environ["WANDB_WATCH"] = wandb.config.wandb_watch


    # Load dataset from the hub
    dataset = load_dataset(wandb.config.dataset_id)

    if wandb.config.debug_mode:
        dataset['train'] = dataset['train'].select(wandb.config.debug_dataset_indices)
        dataset['test'] = dataset['test'].select(wandb.config.debug_dataset_indices)

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")


    # Load model and tokenizer of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(wandb.config.model_id)


    # DATA PREPROCESSING
    # The maximum total input sequence length after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded.
    tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    # The maximum total sequence length for target text after tokenization. 
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")


    def preprocess_function(sample,padding="max_length"):
        # add prefix to the input for t5
        inputs = ["summarize: " + item for item in sample["dialogue"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

    # METRICS
    nltk.download("punkt")

    # Metric
    metric = evaluate.load("rouge")

    # helper function to postprocess text
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # DATA COLLATOR
    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # Set training args
    training_args = Seq2SeqTrainingArguments(
        report_to="wandb",
        run_name=wandb.config.run_name,
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.per_device_train_batch_size,
        per_device_eval_batch_size=wandb.config.per_device_eval_batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        predict_with_generate=True,
        fp16=wandb.config.fp16, # Overflows with fp16
        learning_rate=wandb.config.learning_rate,
        num_train_epochs=wandb.config.num_train_epochs,
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=wandb.config.logging_steps,
        evaluation_strategy=wandb.config.evaluation_strategy,
        save_strategy=wandb.config.save_strategy,
        save_total_limit=wandb.config.save_total_limit,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
    )

    # Create Trainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Do evaluation
    trainer.evaluate()

    # Save our tokenizer and create model card
    # tokenizer.save_pretrained(repository_id)


    # # EVALUATION
    # from transformers import pipeline
    # from random import randrange        

    # # load model and tokenizer from huggingface hub with pipeline
    # summarizer = pipeline("summarization", 
    #                       model="philschmid/flan-t5-base-samsum", 
    #                       device=0)

    # # select a random test sample
    # sample = dataset['test'][randrange(len(dataset["test"]))]
    # print(f"dialogue: \n{sample['dialogue']}\n---------------")

    # # summarize dialogue
    # res = summarizer(sample["dialogue"])

    # print(f"flan-t5-base summary:\n{res[0]['summary_text']}")

if __name__ == "__main__":
    args = parse_args()
    main(args)