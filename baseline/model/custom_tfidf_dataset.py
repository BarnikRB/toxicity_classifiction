# custom_tfidf_dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TfidfDataset(Dataset):
    def __init__(self, file_path, vectorizer=None):
        """
        Dataset class for TF-IDF features.
        If vectorizer is None, returns raw text for fitting.
        If vectorizer is provided, transforms text to TF-IDF features.
        
        Args:
            file_path (str): Path to the TSV data file
            vectorizer: Fitted TfidfVectorizer or None
        """
        # Load data from the .tsv file
        self.data = pd.read_csv(file_path, sep='\t', header=0, quoting=3)
        self.vectorizer = vectorizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get text and label from the dataset
        text = self.data.iloc[idx]['text']  # Assuming "text" is the column name
        label = self.data.iloc[idx]['label']  # Assuming "label" is the column name
        
        # If vectorizer is None, return raw text for fitting
        if self.vectorizer is None:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        # Transform text to TF-IDF vector
        features = self.vectorizer.transform([text]).toarray()
        features_tensor = torch.FloatTensor(features.squeeze())
        
        return {
            'features': features_tensor,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_all_texts(self):
        """Helper method to get all texts for fitting the vectorizer"""
        return self.data['text'].tolist()
