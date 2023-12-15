import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

class TextCustomDataset(Dataset):
    def __init__(self,df,tokenizer,max_length=128) -> None:
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):

        title = self.df['Title'].iloc[index]
        content = self.df['Content'].iloc[index]
        keywords = self.df['Keywords'].iloc[index]
        description = self.df['Description'].iloc[index]
        label = self.df['Label'].iloc[index]

        inputs_title = self.tokenizer(
            title,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        inputs_content = self.tokenizer(
            content,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        if not isinstance(keywords, float):
            inputs_keywords = self.tokenizer(
                keywords,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            inp_key_tensor = inputs_keywords['input_ids']
            attn_key_tensor = inputs_keywords['attention_mask']
        else:
            inp_key_tensor = torch.zeros((1,128))
            attn_key_tensor = torch.zeros((1,128))

        if not isinstance(description, float):
            inputs_desc = self.tokenizer(
                description,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            inp_desc_tensor = inputs_desc['input_ids']
            attn_desc_tensor = inputs_desc['attention_mask']
        else:
            inp_desc_tensor = torch.zeros((1,128))
            attn_desc_tensor = torch.zeros((1,128))

        input_ids = torch.concat([
            inputs_title['input_ids'],
            inputs_content['input_ids'],
            inp_key_tensor,
            inp_desc_tensor
        ],axis=1)

        attn_mask = torch.concat([
            inputs_title['attention_mask'],
            inputs_content['attention_mask'],
            attn_key_tensor,
            attn_desc_tensor
        ],axis=1)

        return {
            'input_ids': input_ids.squeeze().to(self.device),
            'attention_mask': attn_mask.squeeze().to(self.device),
            'label':torch.tensor(label, dtype=torch.long).to(self.device)
        }

    def __len__ (self):
        return len(self.df)

class TextClassificationModel(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-german-cased').to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-5)

    def fine_tune(self,epochs,train_dataset):
        train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                outputs = self.model(input_ids.long(),labels=labels,attention_mask=attention_mask)
                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            print(epoch,epoch_loss)

    def forward(self,batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = self.model(input_ids,attention_mask=attention_mask)
        return outputs

    def eval(self,test_dataset):
        test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True)
        preds = []
        ys = []

        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = self.model(input_ids.long(),labels=labels,attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).tolist()
            preds.append(predicted_labels)
            ys.append(labels.cpu().detach().numpy())
        return accuracy_score(np.concatenate(ys),np.concatenate(preds))

    def save(self):
        self.model.save_pretrained('/content/drive/MyDrive/PretrainedTextModel')

if __name__ == "__main__":
    df = pd.read_csv('/content/drive/MyDrive/GermanFakeNewsProcessed.csv').drop(columns='Images_URLs')
    df['Title'] = df['Title'].apply(lambda x: x.split("'")[1::2]).apply(lambda x: ' '.join(x))
    df['Content'] = df['Content'].apply(lambda x: x.split("'")[1::2]).apply(lambda x: ' '.join(x))
    df['Keywords'] = df['Keywords'].apply(lambda x: x.split("'")[1::2] if not isinstance(x, float) else np.nan).apply(lambda x: ' '.join(x) if not isinstance(x, float) else np.nan)
    df['Description'] = df['Description'].apply(lambda x: x.split("'")[1::2] if not isinstance(x, float) else np.nan).apply(lambda x: ' '.join(x) if not isinstance(x, float) else np.nan)
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

    train_dataset = TextCustomDataset(train_df,tokenizer)
    test_dataset = TextCustomDataset(test_df,tokenizer)

    model = TextClassificationModel()
    prev_acc = model.eval(test_dataset)
    model.fine_tune(5,train_dataset)
    model.save()
    acc = model.eval(test_dataset)
    print('Accuracy: ' + str(prev_acc) + ' -> ' + str(acc))
