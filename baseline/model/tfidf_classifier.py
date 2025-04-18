# tfidf_classifier.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

class TfidfClassifier(nn.Module):
    """
    A PyTorch wrapper around a TF-IDF + classifier pipeline for toxic document classification.
    This serves as a replacement for the MultilingualBertClassifier.
    """
    
    def __init__(self, max_features=10000, min_df=5, ngram_range=(1, 2), 
                 num_classes=2, hidden_dims=None, dropout=0.2, simple=False):
        super(TfidfClassifier, self).__init__()
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        
        # Feature dimension not known until fit
        self.input_dim = None
        self.num_classes = num_classes
        self.simple = simple
        self.dropout_rate = dropout
        
        # Network architecture will be initialized after vectorizer is fit
        self.classifier = None
        self.hidden_dims = hidden_dims or [512]

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        """Load model directly from checkpoint without re-initialization"""
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Create an empty model instance
        model = cls()
        
        # Set the vectorizer
        model.vectorizer = checkpoint['vectorizer']
        model.input_dim = len(model.vectorizer.get_feature_names_out())
        
        # Load the state dict to get the architecture and weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def initialize_classifier(self):
        self.input_dim = len(self.vectorizer.get_feature_names_out())

        # Initialize the classifier architecture
        if self.simple:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.input_dim, self.num_classes)
            )
        else:
            layers = []
            layers.append(nn.Dropout(p=self.dropout_rate))
            layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
            layers.append(nn.ReLU())
            
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Dropout(p=self.dropout_rate))
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                layers.append(nn.ReLU())
                
            layers.append(nn.Dropout(p=self.dropout_rate))
            layers.append(nn.Linear(self.hidden_dims[-1], self.num_classes))
            
            self.classifier = nn.Sequential(*layers)

    def fit_vectorizer(self, texts):
        """Fit the TF-IDF vectorizer and initialize the classifier"""
        self.vectorizer.fit(texts)
        
        self.initialize_classifier()
        
    
    def transform_text(self, texts):
        """Transform raw texts to TF-IDF features"""
        X = self.vectorizer.transform(texts).toarray()
        return torch.FloatTensor(X)
        
    def forward(self, x):
        """Forward pass - x is already TF-IDF transformed features"""
        return self.classifier(x)
    
    def predict(self, texts):
        """Convenience method for direct prediction from raw text"""
        self.eval()
        with torch.no_grad():
            x = self.transform_text(texts)
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
