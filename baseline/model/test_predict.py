import torch
import torch.nn as nn
import json
import argparse
import os
import numpy as np
import pandas as pd
from tfidf_classifier import TfidfClassifier
from custom_tfidf_dataset import TfidfDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from model_train_tfidf import set_seed, create_dataloader, collate_tfidf_batch



def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_model(checkpoint_path, config):
    """
    Load a trained model and its vectorizer from checkpoint
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Initialize the model
    model = TfidfClassifier(
        max_features=config.get("max_features", 10000),
        min_df=config.get("min_df", 5),
        ngram_range=config.get("ngram_range", (1, 2)),
        num_classes=config.get("num_classes", 2),
        hidden_dims=config.get("hidden_dims", [512]),
        dropout=config.get("dropout", 0.2),
        simple=config.get("simple", False)
    )
    
    # Set the vectorizer from checkpoint
    model.vectorizer = checkpoint['vectorizer']
    
    # Initialize the model architecture
    model.input_dim = len(model.vectorizer.get_feature_names_out())
    model.initialize_classifier() # This just initializes the classifier architecture
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def evaluate(model, val_loader, device):
    """
    Evaluate the model on a validation set
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)

            output = model(features)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    return all_preds


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
        print(f"\nEvaluating model with config from: {config_path}")
        
        set_seed(42)
        config = load_config(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the trained model
        model_path = os.path.join(config.get("model_save_dir", "./tfidf_checkpoints"), "checkpoint_best.pt")
        model = load_model(model_path, config)
        model = model.to(device)
        
        # Create dataloader for evaluation
        test_df = pd.read_csv(config.get("test_data"), sep='\t', header=0, quoting=3)

        predictions = []
        with torch.no_grad():
            for t in test_df["text"].values:
                
                features = model.vectorizer.transform([t]).toarray()
                features_tensor = torch.FloatTensor(features.squeeze())

                output = model(features_tensor.to(device))
                preds = torch.argmax(output)
                predictions.append(preds.cpu().numpy())

        test_df["predicted"] = predictions
        test_df.to_csv("test_predictions.tsv", sep='\t', quoting=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF models using configurations from a directory.")
    parser.add_argument('--config_dir', type=str, required=True, help='Path to directory containing config files')
    parser.add_argument('--configs', nargs='*', help='Specific config filenames to use (e.g., config1.json config2.json)')
    args = parser.parse_args()

    main(args.config_dir, args.configs)
