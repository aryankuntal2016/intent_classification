import os, sys
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import pickle

from data_handler import TEXT, LABELS, NUM_INTENTS









MODEL_NAME = "distilbert-base-uncased"



TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

ENCODINGS = TOKENIZER(TEXT, truncation=True, padding=True, max_length=128)



class IntentDataset(Dataset):

	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
		item["labels"] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

TRAINING_DATASET = IntentDataset(ENCODINGS, LABELS)

MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = NUM_INTENTS)

TRAINING_ARGS = TrainingArguments(
    output_dir = './training_results',
    num_train_epochs = 10,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    warmup_steps = 100,
    weight_decay = 0.01,
    # logging_dir = './logs',
    logging_steps = 10,
    no_cuda = True
)

TRAINER = Trainer(
    model = MODEL,
    args = TRAINING_ARGS,
    train_dataset = TRAINING_DATASET
)





