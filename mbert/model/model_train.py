import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertTokenizer
from bert_classifier import MultilingualBertClassifier
from custom_dataset import TextDataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

def create_dataloader(filepath,tokenizer, batch_size=16, max_length=512,shuffle=False):
    dataset = TextDataset(file_path=filepath,tokenizer=tokenizer,max_length=max_length)
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset=dataset,batch_size=batch_size,collate_fn=collate_fn,shuffle=shuffle)
    return loader


def early_stopping(val_loss,best_val_loss,count,patience):
    if val_loss < best_val_loss:
        count = 0
        best_val_loss = val_loss
    else:
        count += 1
    if count >= patience:
        print('Early Stopping Triggered')
        return False, best_val_loss,count
    return True, best_val_loss,count



def train(model,train_loader,val_loader,epochs,optimizer,criterion,patience,model_save_dir,device):
    continue_training = True
    best_val_loss = float('inf')
    counter = 0
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
                print(f'Done {idx+1}')
        running_loss /= len(train_loader)
        print(f'running_loss {running_loss}')
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                input_mask = batch['attention_mask'].to(device)
                
                labels = batch['labels'].to(device)
                output = model(input_ids,input_mask)
                loss = criterion(output,labels)
                val_loss += loss.item()
            val_loss/=len(val_loader)
        continue_training, best_val_loss, counter = early_stopping(val_loss=val_loss, best_val_loss=best_val_loss,count= counter,patience=patience)
        print(f'val_loss: {val_loss}')
        print(counter)
        if continue_training:
            if counter == 0:
                epoch_val_loss_dict[epoch+1] = val_loss
                model_save_path = f'{model_save_dir}/checkpoint_best.pt'
                torch.save(model.state_dict(),model_save_path)
        else:
            break
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}')
        return epoch_val_loss_dict




if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    train_loader = create_dataloader(filepath = './mbert/data/train.tsv',tokenizer=tokenizer, batch_size=32, max_length=512,shuffle=True)
    val_loader = create_dataloader(filepath = './mbert/data/dev.tsv',tokenizer=tokenizer, batch_size=32, max_length=512,shuffle=False)
    model = MultilingualBertClassifier(
        bert_model_name=bert_model,
        num_classes=2,
        freeze_bert=True,
        unfreeze_layers=None,

    ).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    val_losses = train(model=model,train_loader=train_loader,val_loader=val_loader,epochs=50, optimizer=optimizer,criterion=criterion,patience = 5,model_save_dir='./mbert/checkpoints',device=device)
        

