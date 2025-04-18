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
from model_train import set_seed, create_dataloader, create_dataloader_test
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F


def evaluate(model, val_loader, criterion, device):
    """
    function for evaluating model performance on validation set
    returns the prediction as well as probabilities for plotting ROC
    """
    
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, input_mask)  # logits
            loss = criterion(output, labels)
            val_loss += loss.item()

            probs = F.softmax(output, dim=1)  # class probabilities
            class1_probs = probs[:, 1]  # probability of class 1
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(class1_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    print(f'val_loss: {val_loss:.4f}')

    metrics = {
        "f1": {
            "0": f1_score(all_labels, all_preds, pos_label=0, zero_division=0),
            "1": f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        },
        "precision": {
            "0": precision_score(all_labels, all_preds, pos_label=0, zero_division=0),
            "1": precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        },
        "recall": {
            "0": recall_score(all_labels, all_preds, pos_label=0, zero_division=0),
            "1": recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        }
    }

    return all_preds, all_probs, metrics


def evaluate_test(model, val_loader,  device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['attention_mask'].to(device)
            

            output = model(input_ids, input_mask)
            

            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            

    val_loss /= len(val_loader)

    
    return all_preds

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

        
        # val_loader = create_dataloader(
        #     filepath=config["val_data"],
        #     tokenizer=tokenizer,
        #     batch_size=config.get("batch_size", 32),
        #     max_length=config.get("max_length", 512),
        #     shuffle=False
        # )
        test_path = 'D:/toxicity_classifiction/mbert/data/test.tsv'
        test_loader = create_dataloader_test(
            filepath=test_path,
            tokenizer=tokenizer,
            batch_size=config.get("batch_size", 32),
            max_length=config.get("max_length", 512),
            shuffle=False
        )

        model_checkpoint = config.get("model_save_dir", "./checkpoints")
        model_checkpoint = model_checkpoint + '/checkpoint_best.pt'

        model = MultilingualBertClassifier(
            bert_model_name=bert_model,
            num_classes=config.get("num_classes", 2),
            freeze_bert=config.get("freeze_bert", True),
            unfreeze_layers=config.get("unfreeze_layers", None),
            simple=config.get("simple", False),
        )

        model.load_state_dict(torch.load(model_checkpoint))

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        

        # val_pred,val_prob,metrics = evaluate(model=model,val_loader=val_loader,criterion=criterion,device=device)
        # print(metrics)
        test_pred = evaluate_test(model=model,val_loader=test_loader,device=device)
        test_df = pd.read_csv(test_path, sep='\t', header=0, quoting=3)
        # val_df = pd.read_csv(config["val_data"], sep='\t', header=0, quoting=3)
        # val_df['predicted'] = val_pred
        # val_df['probs'] = val_prob
        test_df['predicted'] = test_pred
        test_df = test_df[['id','predicted']]
        test_df.to_csv(f'./mbert/checkpoints/config_final/test_pred.tsv', sep='\t', index=False)
        # val_df.to_csv(f'./mbert/checkpoints/config2/val_pred.tsv', sep='\t', index=False)
        
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models using configurations from a directory.")
    parser.add_argument('--config_dir', type=str, required=True, help='Path to directory containing config files')
    parser.add_argument('--configs', nargs='*', help='Specific config filenames to use (e.g., config1.json config2.json)')
    args = parser.parse_args()

    main(args.config_dir, args.configs)
