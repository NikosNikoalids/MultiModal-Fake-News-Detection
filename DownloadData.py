#########################################################################
#  Check that the given URLs are still up and then download the         #
#  Content, Title, Metadata, Images_URLs and adds the labels to the CSV #
#########################################################################

import json
import requests
import pickle
import csv
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def ParseData(url):
    
    try:
        response = requests.get(url)
    except:
        return None
    
    if response.status_code == 200:
        try:
            soup = BeautifulSoup(response.text,'html.parser')
        except:
            return None
        ### Title
        try:
            title = soup.title.text.strip()
        except:
            title = None
        
        ### Content
        try:
            text = '\n'.join([p.text.strip() for p in soup.find_all('p')])
        except:
            text = None
        
        ### Metadata
        try:
            metadata = {}
            meta_tags = soup.head.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name')
                content = tag.get('content')
                if name and content:
                    metadata[name.lower()] = content
        except:
            metadata = None

        ### Images
        try:
            img_tags = soup.find_all('img')
            image_urls = [urljoin(url, img['src']) for img in img_tags if 'src' in img.attrs]
        except:
            image_urls = None
        
        return {
            'Title': title,
            'Content': text,
            'Metadata': metadata,
            'Images_URLs': image_urls
        }
    else:
        return None


if __name__ == "__main__":
    
    with open("GermanFakeNC.json") as dataJSON:
        data = json.load(dataJSON)
    
    ExtractedData = list()
    for item in tqdm(data,'Extracting Data From URLs'):
        item_data = ParseData(item['URL'])
        if item_data != None:
            item_data.update({
                'Overall Rating': item['Overall_Rating']
            })
            ExtractedData.append(item_data)
    
    print("Exporting to CSV...")
    csv_file = 'GermanFakeNewsDataset.csv'
    headers = ['Title','Content','Metadata','Images_URLs','Overall Rating']
    with open(csv_file, 'w+', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=headers)
        csv_writer.writeheader()
        csv_writer.writerows(ExtractedData)
    
    percentage = round((len(ExtractedData) / len(data))*100,4)
    print("Percentage of valid entries in JSON: {percentage} %".format(percentage=percentage))