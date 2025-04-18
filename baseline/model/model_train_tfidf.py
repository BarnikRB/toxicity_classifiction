# model_train_tfidf.py
import torch
import torch.nn as nn
import json
import argparse
import os
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from tfidf_classifier import TfidfClassifier
from custom_tfidf_dataset import TfidfDataset

def set_seed(seed: int = 42):
    """
    Sets seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def collate_tfidf_batch(batch):
    """
    Custom collate function for TF-IDF features
    """
    if 'text' in batch[0]:
        # For raw text mode (during vectorizer fitting)
        texts = [item['text'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'text': texts,
            'labels': labels
        }
    else:
        # For feature mode
        features = torch.stack([item['features'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        return {
            'features': features,
            'labels': labels
        }

def create_dataloader(filepath, vectorizer=None, batch_size=16, shuffle=False):
    """
    Create a DataLoader with TF-IDF features
    """
    dataset = TfidfDataset(file_path=filepath, vectorizer=vectorizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_tfidf_batch,
        shuffle=shuffle
    )
    return loader, dataset

def early_stopping(val_loss, best_val_loss, count, patience):
    """
    Early stopping logic
    """
    if val_loss < best_val_loss:
        count = 0
        best_val_loss = val_loss
    else:
        count += 1
    if count >= patience:
        print('Early Stopping Triggered')
        return False, best_val_loss, count
    return True, best_val_loss, count

def train(model, train_loader, val_loader, epochs, optimizer, criterion, patience, model_save_dir, device):
    """
    Training function
    """
    continue_training = True
    best_val_loss = float('inf')
    counter = 0
    epoch_val_loss_dict = {}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (idx+1) % 100 == 0:
                print(f'Done {idx+1} batches')
                
        running_loss /= len(train_loader)
        print(f'running_loss {running_loss}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                output = model(features)
                loss = criterion(output, labels)
                val_loss += loss.item()
                
            val_loss /= len(val_loader)
            
        continue_training, best_val_loss, counter = early_stopping(
            val_loss=val_loss, 
            best_val_loss=best_val_loss,
            count=counter,
            patience=patience
        )
        
        print(f'val_loss: {val_loss}')
        print(f'counter: {counter}')
        
        if continue_training:
            if counter == 0:
                epoch_val_loss_dict[epoch+1] = val_loss
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                model_save_path = f'{model_save_dir}/checkpoint_best.pt'
                
                # Save both the model and the vectorizer
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vectorizer': model.vectorizer
                }, model_save_path)
        else:
            break
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    return epoch_val_loss_dict

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

    for idx, config_path in enumerate(config_paths):
        print(f"\nLoading config from: {config_path}")
        
        set_seed(42)
        config = load_config(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_and_val = config.get("train_and_val", False)
        if train_and_val:
            print("Training on combined train and validation datasets")
        
        # First create dataloader(s) with raw text for fitting the vectorizer
        temp_train_loader, train_dataset = create_dataloader(
            filepath=config["train_data"],
            vectorizer=None,  # No vectorizer yet
            batch_size=config.get("batch_size", 32),
            shuffle=False
        )
        
        # If train_and_val is True, also load validation dataset for vectorizer fitting
        if train_and_val:
            temp_val_loader, val_dataset = create_dataloader(
                filepath=config["val_data"],
                vectorizer=None,
                batch_size=config.get("batch_size", 32),
                shuffle=False
            )
        
        # Initialize the model
        model = TfidfClassifier(
            max_features=config.get("max_features", 10000),
            min_df=config.get("min_df", 5),
            ngram_range=tuple(config.get("ngram_range", (1, 2))),
            num_classes=config.get("num_classes", 2),
            hidden_dims=config.get("hidden_dims", [512]),
            dropout=config.get("dropout", 0.2),
            simple=config.get("simple", False)
        )
        
        # Fit the vectorizer on training data (and validation data if enabled)
        print("Fitting TF-IDF vectorizer...")
        train_texts = train_dataset.get_all_texts()
        
        if train_and_val:
            val_texts = val_dataset.get_all_texts()
            # Combine texts for fitting
            all_texts = train_texts + val_texts
            model.fit_vectorizer(all_texts)
            print(f"Vectorizer fitted with {model.input_dim} features on {len(all_texts)} documents")
        else:
            model.fit_vectorizer(train_texts)
            print(f"Vectorizer fitted with {model.input_dim} features on {len(train_texts)} documents")
        
        # Create proper loader with the fitted vectorizer for training
        train_loader, _ = create_dataloader(
            filepath=config["train_data"],
            vectorizer=model.vectorizer,
            batch_size=config.get("batch_size", 32),
            shuffle=True
        )
        
        # Create loader for validation data
        val_loader, _ = create_dataloader(
            filepath=config["val_data"],
            vectorizer=model.vectorizer,
            batch_size=config.get("batch_size", 32),
            shuffle=False
        )
        
        # If training on combined data, create a new combined loader
        if train_and_val:
            # Create a second training loader for the validation data
            val_train_loader, _ = create_dataloader(
                filepath=config["val_data"],
                vectorizer=model.vectorizer,
                batch_size=config.get("batch_size", 32),
                shuffle=True  # Shuffle for training
            )
            
            # Combine the two dataloaders for training
            # Option 1: Alternate between the two loaders
            combined_loader = train_loader
            if val_train_loader.dataset:
                # Simply use validation data as the validation set too
                # This isn't ideal for monitoring overfitting, but it works
                # when you want to use all available data for the final model
                pass
        
        # Move model to device
        model = model.to(device)
        
        # Setup criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Train the model - if train_and_val is True, use combined data
        if train_and_val:
            # For the final model training, we'll use both train and validation data
            # To monitor progress, we'll still validate on the validation set
            # This isn't ideal for preventing overfitting, but it maximizes the data use
            
            # First, train on training data only, validating on validation data
            print("Step 1: Training on training data with validation monitoring")
            val_losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.get("epochs", 50),
                optimizer=optimizer,
                criterion=criterion,
                patience=config.get("patience", 5),
                model_save_dir=config.get("model_save_dir", "./tfidf_checkpoints"),
                device=device
            )
            
            # Now finetune on combined data
            # Create a combined dataset
            print("Step 2: Fine-tuning on combined train+validation data")
            
            # Create a combined dataset
            train_df = pd.read_csv(config["train_data"], sep='\t', header=0, quoting=3)
            val_df = pd.read_csv(config["val_data"], sep='\t', header=0, quoting=3)
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
            
            # Save combined data to a temporary file
            temp_file = os.path.join(os.path.dirname(config["train_data"]), "temp_combined.tsv")
            combined_df.to_csv(temp_file, sep='\t', index=False, quoting=3)
            
            # Create dataloaders for the combined data
            combined_loader, _ = create_dataloader(
                filepath=temp_file,
                vectorizer=model.vectorizer,
                batch_size=config.get("batch_size", 32),
                shuffle=True
            )
            
            # Train on combined data with a smaller patience
            # Use a smaller subset for validation to monitor progress
            val_losses_combined = train(
                model=model,
                train_loader=combined_loader,
                val_loader=val_loader,  # Still use validation data for monitoring
                epochs=config.get("final_epochs", 10),  # Use fewer epochs for fine-tuning
                optimizer=optimizer,
                criterion=criterion,
                patience=config.get("final_patience", 3),  # Shorter patience for fine-tuning
                model_save_dir=config.get("model_save_dir", "./tfidf_checkpoints") + "_combined",
                device=device
            )
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            print(f'Done {config_path}. Initial training stopped at epoch {min(val_losses, key=val_losses.get)}')
            print(f'Fine-tuning on combined data stopped at epoch {min(val_losses_combined, key=val_losses_combined.get)}')
        else:
            # Regular training on training data only
            val_losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.get("epochs", 50),
                optimizer=optimizer,
                criterion=criterion,
                patience=config.get("patience", 5),
                model_save_dir=config.get("model_save_dir", "./tfidf_checkpoints"),
                device=device
            )
            
            print(f'Done {config_path}. Training stopped at epoch {min(val_losses, key=val_losses.get)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TF-IDF models using configurations from a directory.")
    parser.add_argument('--config_dir', type=str, required=True, help='Path to directory containing config files')
    parser.add_argument('--configs', nargs='*', help='Specific config filenames to use (e.g., config1.json config2.json)')
    args = parser.parse_args()

    main(args.config_dir, args.configs)
