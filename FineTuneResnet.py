import torch
import pickle
import pandas as pd
from torch import nn
import numpy as np
from torch import optim
import torchvision.models as models
from sklearn.metrics import accuracy_score

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

class ResentFineTuning(nn.Module):
  def __init__(self):
    super(ResentFineTuning,self).__init__()
    self.resnet = models.resnet18()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_ftrs, 2)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.AdamW(self.resnet.parameters(), lr=1e-4)
    self.resnet.to(self.device)

  def fine_tune(self,epochs,train_dataloader):
    for epoch in range(epochs):
      epoch_loss = 0.0
      for X, y in train_dataloader:
        self.optimizer.zero_grad()
        outputs = self.resnet(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()
      print(epoch,epoch_loss)

  def eval(self,test_dataloader):
    ys, preds = list(), list()
    for X, y in test_dataloader:
      self.optimizer.zero_grad()
      outputs = self.resnet(X)
      probabilities = torch.nn.functional.softmax(outputs, dim=1)
      predicted_labels = torch.argmax(probabilities, dim=1).tolist()
      preds.append(predicted_labels)
      ys.append(y.cpu().detach().numpy())
    print(accuracy_score(np.concatenate(ys),np.concatenate(preds)))
  def save(self):
    torch.save(self.resnet.state_dict(), '/content/drive/MyDrive/ResnetTuned.pth')

if __name__ == "__main__":
  df = pd.read_csv("/content/drive/MyDrive/GermanFakeNewsProcessed.csv")[['ArticleID','Label']]
  with open("/content/drive/MyDrive/ImagesDatasetNew.pkl",'rb') as f:
    imgs_data = pickle.load(f)

  dataset = ImagesDataset(imgs_data,df)
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
  model = ResentFineTuning()
  model.eval(test_loader)
  model.fine_tune(10,train_loader)
  model.eval(test_loader)
  model.save()