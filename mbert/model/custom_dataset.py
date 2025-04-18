import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import pandas as pd

# Custom Dataset class to handle tokenization
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Load data from the .tsv file
        self.data = pd.read_csv(file_path, sep='\t', header=0, quoting=3)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get text and label from the dataset
        text = self.data.iloc[idx]['text']  # Assuming "text" is the column name
        label = self.data.iloc[idx]['label']  # Assuming "label" is the column name

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return input_ids, attention_mask, and label
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }




class TextDatasetTest(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        # Load data from the .tsv file
        self.data = pd.read_csv(file_path, sep='\t', header=0, quoting=3)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get text and label from the dataset
        text = self.data.iloc[idx]['text']  # Assuming "text" is the column name
          # Assuming "label" is the column name

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Return input_ids, attention_mask, and label
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            
        }