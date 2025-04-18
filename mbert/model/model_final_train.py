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

def set_seed(seed: int = 42):
    """
    Sets seeds for reproducibility across Python, NumPy, and PyTorch.
    Leaves cuDNN settings unchanged to avoid impacting performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_dataloader(filepath,tokenizer, batch_size=16, max_length=512,shuffle=False):
    dataset = TextDataset(file_path=filepath,tokenizer=tokenizer,max_length=max_length)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset=dataset,batch_size=batch_size,collate_fn=collate_fn,shuffle=shuffle)
    return loader


def train(model,train_loader,epochs,optimizer,criterion,model_save_dir,device):
    # Same as model_train.py but with no early stopping based on validation error since the training and validation sets are combined
    epoch_val_loss_dict = {}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for idx,batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            input_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            output = model(input_id = input_ids,input_attention_mask = input_mask)
            loss = criterion(output,labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (idx+1)%100 == 0:
                print(f'Done {idx+1} batches')
        running_loss /= len(train_loader)
        
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss:.4f}')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = f'{model_save_dir}/checkpoint_best.pt'
    torch.save(model.state_dict(),model_save_path)
    


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

        train_loader = create_dataloader(
            filepath=config["train_data"],
            tokenizer=tokenizer,
            batch_size=config.get("batch_size", 32),
            max_length=config.get("max_length", 512),
            shuffle=True
        )
        
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

        train(
            model=model,
            train_loader=train_loader,
            epochs=config.get("epochs", 50),
            optimizer=optimizer,
            criterion=criterion,
            model_save_dir=config.get("model_save_dir", "./checkpoints"),
            device=device
        )

        print(f'Done {config_path}')



# python model_train.py --config_dir ./configs --configs config1.json config2.json
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models using configurations from a directory.")
    parser.add_argument('--config_dir', type=str, required=True, help='Path to directory containing config files')
    parser.add_argument('--configs', nargs='*', help='Specific config filenames to use (e.g., config1.json config2.json)')
    args = parser.parse_args()

    main(args.config_dir, args.configs)