import torch
import torch.nn as nn
import json
import argparse
import os
from transformers import BertModel, BertTokenizer
from bert_classifier import MultilingualBertClassifier
from custom_dataset import TextDataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import numpy as np 
import random
from model_train import set_seed, create_dataloader

from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, input_mask)
            loss = criterion(output, labels)
            val_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)

    print(f'val_loss: {val_loss}')

    # Compute metrics for each class
    metrics = {
        "f1": {
            "0": f1_score(all_labels, all_preds, pos_label=0),
            "1": f1_score(all_labels, all_preds, pos_label=1)
        },
        "precision": {
            "0": precision_score(all_labels, all_preds, pos_label=0),
            "1": precision_score(all_labels, all_preds, pos_label=1)
        },
        "recall": {
            "0": recall_score(all_labels, all_preds, pos_label=0),
            "1": recall_score(all_labels, all_preds, pos_label=1)
        }
    }

    return all_preds, metrics

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main(config_dir, config_files):
    if config_files:
        config_paths = [os.path.join(config_dir, fname) for fname in config_files]
    else:
        # Use all .json files in the directory
        config_paths = [
            os.path.join(config_dir, fname)
            for fname in os.listdir(config_dir)
            if fname.endswith('.json')
        ]

    if not config_paths:
        raise ValueError("No configuration files found.")

    for idx,config_path in enumerate(config_paths):
        
        
        print(f"\nLoading config from: {config_path}")
        
        set_seed(42)
        config = load_config(config_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model = config.get("bert_model", "bert-base-multilingual-cased")
        tokenizer = BertTokenizer.from_pretrained(bert_model)

        
        val_loader = create_dataloader(
            filepath=config["val_data"],
            tokenizer=tokenizer,
            batch_size=config.get("batch_size", 32),
            max_length=config.get("max_length", 512),
            shuffle=False
        )

        model

        model = MultilingualBertClassifier(
            bert_model_name=bert_model,
            num_classes=config.get("num_classes", 2),
            freeze_bert=config.get("freeze_bert", True),
            unfreeze_layers=config.get("unfreeze_layers", None),
            simple=config.get("simple", False),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            [{'params':model.bert.parameters(),'lr':config.get("lr_bert", 2e-5)},
             {'params':model.classifier.parameters(),'lr':config.get("lr_classifier", 1e-3)},
             ],
            weight_decay=config.get("weight_decay", 0.01)
        )

        val_losses = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.get("epochs", 50),
            optimizer=optimizer,
            criterion=criterion,
            patience=config.get("patience", 5),
            model_save_dir=config.get("model_save_dir", "./checkpoints"),
            device=device
        )

        print(f'Done {config_path}. Training stopped at epoch {min(val_losses,key=val_losses.get)}')
