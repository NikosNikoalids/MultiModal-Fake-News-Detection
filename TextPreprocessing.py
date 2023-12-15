import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
import langid
import spacy
import de_core_news_md
import nltk

nlp = de_core_news_md.load()
stop_words = set(stopwords.words("german"))
lemmatizer = WordNetLemmatizer()

nltk.download('averaged_perceptron_tagger')
nltk.download("gutenberg")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('words')

def RemoveNonGerman(tokens):
  token_languages = [langid.classify(token)[0] for token in tokens]
  german_tokens = [token for token, lang in zip(tokens, token_languages) if lang == 'de']
  return german_tokens

def SeriesTextPreprocessing(dfSeries):
  ### Lower
  dfSeries = dfSeries.str.lower()
  ### Remove URLs
  dfSeries = dfSeries.apply(lambda x: re.split('https:\/\/.*', str(x))[0])
  ### Remove Non-ASCII
  dfSeries = dfSeries.apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))
  ### Remove Punctuation and Special Characters
  dfSeries = dfSeries.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
  ### Remove Numbers
  dfSeries = dfSeries.apply(lambda x: re.sub(r'\d+', '', x))
  ### Remove \n
  dfSeries = dfSeries.apply(lambda x: re.sub(r'\n', '', x))
  ### Tokenize
  dfSeries = dfSeries.apply(word_tokenize)
  ### Remove stopwords
  dfSeries = dfSeries.apply(lambda x: [word for word in x if word not in stop_words])
  ### Lemmatize
  dfSeries = dfSeries.apply(lambda x: [lemmatizer.lemmatize(word, pos='v') for word in x])
  ### Remove non-German Words
  dfSeries = dfSeries.apply(RemoveNonGerman)

  return dfSeries


if __name__ == "__main__":

    df = pd.read_csv("GermanFakeNewsDataset.csv").reset_index().rename(columns={'index':'ArticleID'})
    df['Metadata'] = df['Metadata'].apply(eval)
    df['Label'] = np.where(df['Overall Rating'] > 0.5,1,0)
    df['Title'] = SeriesTextPreprocessing(df["Title"])
    df['Content'] = SeriesTextPreprocessing(df["Content"])
    df['TitleLength'] = df['Title'].apply(lambda x: len(x))
    df['ContentLength'] = df['Content'].apply(lambda x: len(x))
    df['Description'] = SeriesTextPreprocessing(df['Metadata'].apply(lambda x: x.get("description", np.nan)))
    df['Keywords'] = SeriesTextPreprocessing(df['Metadata'].apply(lambda x: x.get("keywords", np.nan)))
    df['Keywords'] = df['Keywords'].apply(lambda x: x if len(x) > 0  else np.nan)
    df['Description'] = df['Description'].apply(lambda x: x if len(x) > 0  else np.nan)
    df = df[df['ContentLength'] != 0]
    df = df[df['TitleLength'] != 0]
    df.drop(columns=['ContentLength','TitleLength','Overall Rating','Metadata'],inplace=True)
    df.to_csv('GermanFakeNewsProcessed.csv',index=False)