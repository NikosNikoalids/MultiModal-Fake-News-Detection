import pandas as pd
import requests
from io import BytesIO
import os
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import pickle


def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return None
    try:
      image = Image.open(BytesIO(response.content)).convert('RGB')
      return image
    except:
      return None

def load_and_preprocess_image(image, model_input_size=(224, 224), normalization_mean=[0.485, 0.456, 0.406], normalization_std=[0.229, 0.224, 0.225]):
    transform = transforms.Compose([
        transforms.Resize(model_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization_mean, std=normalization_std)
    ])

    return transform(image).unsqueeze(0)

if __name__ == "__main__":
    data = pd.read_csv("/content/drive/MyDrive/GermanFakeNewsProcessed.csv")[['ArticleID','Images_URLs']]
    images_dict = dict()
    
    for index, row in data.iterrows():
        print("Processing " + str(index))
        images_urls = eval(row['Images_URLs'])
        images_tensors = list()
        for url in tqdm(images_urls):
            img = download_image(url)
            if img:
                images_tensors.append(load_and_preprocess_image(img))
        if len(images_tensors) == 0:
            images_tensors.append(torch.zeros((1,3,224,224)))
        images_dict[row['ArticleID']] = torch.concat(images_tensors,dim=0)
    
    with open('/content/drive/MyDrive/ImagesDataset.pkl', 'wb') as f:
        pickle.dump(images_dict, f)