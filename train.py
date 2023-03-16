'''
    This script is a simple example of how to use Weights & Biases Launch
    to fine-tune a T5 model.

    Credits:
    This script is based on an example script from Philipp Schmid
      - blog: post:https://www.philschmid.de/fine-tune-flan-t5
      - code: https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/flan-t5-samsum-summarization.ipynb
'''

import os
import wandb
import torch
import argparse

import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default='launch-demo-flan-t5',
        required=False,
        help="Weights & Biases Project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        required=False,
        help="Weights & Biases Team or Username to log to",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        required=False,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="samsum",
        required=False,
        help="Dataset from Hugging Face Datasets to use for training",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/flan-t5-base",
        required=False,
        help="Model to use for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        required=False,
        help="Learning rate to use for training",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        required=False,
        help="Learning rate warmup steps",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        required=False,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        required=False,
        help="Number of steps to accumulate gradients for",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        required=False,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        required=False,
        help="Evaluation batch size per device",
    )  
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Use fp16 during training",
    )   
    parser.add_argument(
        "--bf16",
        default=False,
        action="store_true",
        help="Use bf16 training",
    )    
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        required=False,
        help="Save strategy to use for training. Can be 'steps', 'epoch' or 'no'",
    )
    parser.add_argument(
        "--save_steps",
        type=str,
        default=200,
        required=False,
        help="If save_strategy is 'steps', save checkpoint every save_steps",
    )
    parser.add_argument(
        "--wandb_log_model",
        type=str,
        default="checkpoint",
        required=False,
        help="Log model to Weights & Biases. Can be 'checkpoint', 'best' or 'all'",
    )
    parser.add_argument(
        "--log_code_to_wandb_job_only",
        default=False,
        action="store_true",
        help="Only log the code to a Weights & Biases Job and then exit the script. Useful for registering a Job",
    )    
    parser.add_argument(
        "--debug_mode",
        default=False,
        action="store_true",
        help="Run training in debug mode wtih a tiny dataset",
    )    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

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

 # helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

# Metrics
nltk.download("punkt")
metric = evaluate.load("rouge")

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

def main(args):
    config = {
        # HF Trainer arguments (except for those in `args`)
        "logging_steps":5,
        "evaluation_strategy":"steps", # "epoch",
        "eval_steps":100,
        "save_total_limit":2,
        "output_dir":f"{args.model_id.split('/')[1]}-{args.dataset_id}",

        # Trainer wandb settings
        "wandb_watch":'false',
    
        # Debugging
        "debug_dataset_indices":list(range(128)),
    }   

    # Start a Weights & Biases run and log the config
    config = {**vars(args), **config}
    run = wandb.init(
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            project=args.wandb_project,
            config=config)
    
    # Set all args to use the config defined in the Weights & Biases run
    args = run.config

    # Log the training script to Weighs & Biases for Launch to re-use
    run.log_code()
    
    # If we only want to log the code as a W&B Job, we can finish the run here
    if args.log_code_to_wandb_job_only: 
        print("Finising the wandb run and exiting...")
        run.finish()
        exit()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")

    # Set environment variables for HF Trainer's wandb logging
    os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
    os.environ["WANDB_WATCH"] = args.wandb_watch

    # Load dataset from the hub
    dataset = load_dataset(args.dataset_id)

    if args.debug_mode:
        dataset['train'] = dataset['train'].select(args.debug_dataset_indices)
        dataset['test'] = dataset['test'].select(args.debug_dataset_indices)

    # Log the size our the dataset to Weights & Biases
    run.config["train_dataset_size"] = len(dataset['train'])
    run.config["test_dataset_size"] = len(dataset['test'])
    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")


    # Load model and tokenizer of FLAN-t5-base
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

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

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
    print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

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
        run_name=args.wandb_run_name,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        fp16=args.fp16,
        bf16=args.bf16,
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=5,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
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