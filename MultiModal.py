import torch
import pickle
import pandas as pd
from torch import nn
import numpy as np
from torch import optim
import torchvision.models as models
from sklearn.metrics import accuracy_score,confusion_matrix
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class ImagesDataset(torch.utils.data.Dataset):
  def __init__(self,imgs_data,df):
    super().__init__()
    self.df = df
    self.imgs_data = imgs_data
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def __len__(self):
    return len(self.df)
  def __getitem__(self,idx):
    id = self.df.iloc[idx]['ArticleID']
    label = self.df.iloc[idx]['Label']
    feature = self.imgs_data[id]

    return feature.reshape(3,224,-1).to(self.device), torch.tensor(label).to(self.device)

class TextCustomDataset(torch.utils.data.Dataset):
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

class MultiModalDataset(torch.utils.data.Dataset):
  def __init__(self,imgs_data,df,tokenizer):
    super().__init__()
    self.imgs_dataset = ImagesDataset(imgs_data,df)
    self.text_dataset = TextCustomDataset(df,tokenizer)
    self.N = len(df)
  def __len__(self):
    return self.N
  def __getitem__(self,idx):
    X_img, y_img = self.imgs_dataset[idx]
    batch = self.text_dataset[idx]
    return batch, X_img

class MultimodalMOE(nn.Module):
    def __init__(self):
        super(MultimodalMOE, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained('/content/drive/MyDrive/PretrainedTextModel')

        for param in self.bert.parameters():
            param.requires_grad = False
        self.text_fc = nn.Linear(2, 2)

        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        self.resnet.load_state_dict(torch.load('/content/drive/MyDrive/ResnetTuned.pth'))
        
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.image_fc = nn.Linear(2, 2)

        self.expert_gate = nn.Sequential(
            nn.Linear(2*2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Softmax(dim=1)
        )

        self.expert_mlp = nn.Sequential(
            nn.Linear(2*2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, text_input, image_input):

        input_ids = text_input['input_ids'].reshape(1,-1).long()
        attention_mask = text_input['attention_mask'].reshape(1,-1)
        labels = text_input['label']

        text_embedding = self.bert(input_ids,attention_mask=attention_mask)
        text_output = self.text_fc(text_embedding.logits)

        image_embedded = self.resnet(image_input.reshape(1,3,224,-1))
        image_output = self.image_fc(image_embedded)

        combined_features = torch.cat((text_output, image_output), dim=1)

        expert_gate_weights = self.expert_gate(combined_features)

        expert_output = expert_gate_weights[:, :, None] * combined_features[:, None, :]

        expert_output = expert_output.sum(dim=0)

        output = self.expert_mlp(expert_output)
        return output

def train(train_loader,epochs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = MultimodalMOE().to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.BCELoss()

  for epoch in range(epochs):
    epoch_loss = 0.0
    for batch, img in train_loader:
      outputs = model(batch,img)
      y = batch['label']
      loss = criterion(outputs.squeeze(0),y.float())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    print(epoch_loss)
  return model

def eval(model,test_loader):
  ys, preds = list(), list()
  for batch, img in test_loader:
    outputs = model(batch,img)
    y = batch['label']
    ys.append(y.cpu().detach().numpy())
    preds.append(outputs.squeeze(0).cpu().detach().numpy())

  ys = np.concatenate(ys)
  preds = np.concatenate(preds)
  preds = np.where(preds >= 0.5,1,0)
  print(accuracy_score(ys,preds))
  cm = confusion_matrix(ys, preds)
  sns.heatmap(cm, annot=True)
  plt.show()

if __name__ == "__main__":
  df = pd.read_csv("/content/drive/MyDrive/GermanFakeNewsProcessed.csv")
  df['Title'] = df['Title'].apply(lambda x: x.split("'")[1::2]).apply(lambda x: ' '.join(x))
  df['Content'] = df['Content'].apply(lambda x: x.split("'")[1::2]).apply(lambda x: ' '.join(x))
  df['Keywords'] = df['Keywords'].apply(lambda x: x.split("'")[1::2] if not isinstance(x, float) else np.nan).apply(lambda x: ' '.join(x) if not isinstance(x, float) else np.nan)
  df['Description'] = df['Description'].apply(lambda x: x.split("'")[1::2] if not isinstance(x, float) else np.nan).apply(lambda x: ' '.join(x) if not isinstance(x, float) else np.nan)

  with open("/content/drive/MyDrive/ImagesDatasetNew.pkl",'rb') as f:
    imgs_data = pickle.load(f)
  tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

  dataset = MultiModalDataset(imgs_data,df,tokenizer)
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=True)
  model = train(train_loader,10)
  eval(model,test_loader)