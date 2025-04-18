import torch
import torch.nn as nn
import transformers
from transformers import BertModel


class MultilingualBertClassifier(nn.Module):
    
    def __init__(self, bert_model_name, num_classes = 2, freeze_bert=True, unfreeze_layers=['encoder.layer.11'],dropout1= 0.2,dropout2= 0.2,simple = False):
        super(MultilingualBertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            if unfreeze_layers:
                for name, param in self.bert.named_parameters():
                    if any(layer in name for layer in unfreeze_layers):
                        param.requires_grad = True
        self.classifier  = None
        if not simple: 
            self.classifier  =  nn.Sequential(
                nn.Dropout(p=dropout1),
                nn.Linear(self.bert.config.hidden_size,512),
                nn.ReLU(),
                nn.Dropout(p=dropout2),
                nn.Linear(512,num_classes)

            )
        else:
            self.classifier  =  nn.Sequential(
                nn.Dropout(p=dropout1),
                nn.Linear(self.bert.config.hidden_size,num_classes),


            )

       
        


    def forward(self,input_id,input_attention_mask):
        output = self.bert(input_id,attention_mask=input_attention_mask)
        pooled_output = output.pooler_output
        pooled_output = self.classifier(pooled_output)

        return pooled_output
