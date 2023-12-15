
# MultiModal Fake News-Detection

The objective of this project is to identify articles containing false information by leveraging diverse data sources. The initial dataset includes article URLs along with corresponding labels ranging from 0.1 to 1.0. For the project's purposes, articles with labels exceeding 0.5 are considered to be indicative of fake news.


## Download Data

Scripts: DownloadData.py

Given that the original dataset only comprises article URLs and their associated labels, it becomes necessary to generate a new dataset containing information suitable for input into our model. To achieve this, we proceed with the following steps:

1. Check if URL is up.
2. If it is up we collect the content, title , URLs for images and metadata using Python's BeautifulSoup.
3. We dump the above data and the labels to CSV file.
## Process Data

Scripts: CreateImageDataset.py, TextPreprocessing.py

Once the CSV is generated, the next step involves preprocessing the data to transform it into a format suitable for input into our model.

### Text

1. Remove URLs.
2. Convert to lowercase.
3. Remove non-ASCII characters.
4. Remove punctuation and other special characters.
5. Remove numbers
6. Remove \n
7. Tokenize
8. Remove stop words
9. Remove non-German words
10. Lemmatize

### Images

1. Download Images
2. Transform images to tensors (3,224,224)
3. If there are more than one concat them.
4. If there are no images we fill it with a zeros tensor.
## Fine tune models

Scripts: FineTuneBERT.py, FineTuneResnet.py

### Text

For text we fine tune a transformer's BERT model for classification.

The accuracy after the tuning increases from 47.2% to 66.7%.

### Images

For the images we fine tune pytorch's Resnet18 model.

The accuracy after the tuning increases from 44.4% to 75%.
## Multi-modal

Scripts: MultiModal.py

To integrate the fine-tuned models, we employ a Mixture-Of-Experts architecture. During the training phase, we adopt a strategy where the BERT and ResNet models are frozen, and the optimization process exclusively targets the parameters of the Mixture-Of-Experts component.

The final accuracy after 10 epochs is 97.2%.
