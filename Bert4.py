# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:27:25 2024

@author: mail2
"""

from keybert import KeyBERT
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import itertools
import re
from transformers import BertTokenizer, BertModel
import torch
from textblob import TextBlob

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def get_bert_embedding(text):
    # Encode text
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Get BERT output, output is tuple where first item is last hidden state
    with torch.no_grad():
        output = model(**encoded_input)
    # We'll take mean of embeddings across the input sequence for simplicity
    embeddings = output.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()


df = pd.read_csv("input.csv", header=0, index_col=None, sep=',')

print(df)

# Apply function to get embeddings
df['bert_embedding'] = df['MsgBody'].apply(get_bert_embedding)
df['sentiment_polarity'], df['sentiment_subjectivity'] = zip(*df['MsgBody'].map(get_sentiment))

# default stopwords
stop_words = nltk.corpus.stopwords.words('english')
docs = []
# creating document from the series of articles(text)
for i in df['MsgBody']:
    text = word_tokenize(str(i))
    sentence = []
    for words in text:
        if words not in stop_words:
            sentence.append(words)
    intern = ' '.join(sentence)
    docs.append(intern)

texts = []
# removing ’ from the text
for i in df['MsgBody']:
    i = re.sub(r"[.’]", '', str(i))
    text = word_tokenize(str(i))
    texts.append(text)

# word counter in document
cnts = list(itertools.chain(*texts))
a = Counter(cnts)
print(cnts)

cnt = pd.DataFrame.from_dict(a, orient='index').reset_index()
cnt.rename(columns={'index': 'words', 0: 'counts'}, inplace=True)

# custom common stop words found during the process
new_stopwords = ["read", "dec", "family", "adventure", "new", "year", "wednesday", "monday", "fun", "run", "review",
                 "van", "hit", "find", "long", "plenty", "micheal", "birthday", "upcoming", "staff", "opportunity",
                 "biotech","dietary","unemployed","woolhandle","deforestation","ethiopian","meateate","agrifoodtech",
                 "sprout","woollen"]

# extending the stop word list with custom list
stop_words.extend(new_stopwords)
# initialising kbert for topic modelling
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(docs)
tfidf = []

# creating tdidf scores for the generated words
for i in keywords:
    print(i)
    y = [ele[0] for ele in i if ele[0] not in stop_words]
    mid = ' '.join(y)
    tfidf.append(mid)

tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(tfidf)
# Update the deprecated method call here
df1 = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names_out(), columns=["TF-IDF"])
df1 = df1.sort_values('TF-IDF', ascending=False)
df1['words'] = df1.index
df1.reset_index(drop=True, inplace=True)
print(df1)
print(cnt)

sent_list = []
words_list = []

# finding all the sentiment scores for the keywords
for i in range(len(df)):
    tokens = word_tokenize(str(df.iloc[i]['MsgBody']))
    words_list.append(tokens)
sent_df = pd.DataFrame(list(zip(sent_list, words_list)),columns=['words'])
sent_dict = dict(sent_df.values)

# merging the new tdidf dataframe with the keywords
merge = pd.merge(df1, cnt, how="inner", on="words")

merge['bert_score'] = df['bert_embedding']
merge['sentiment_polarity'] = df['sentiment_polarity']
merge['sentiment_subjectivity'] = df['sentiment_subjectivity']

# writing csv file
merge.to_csv('merge.csv', encoding='utf-8', header=True, sep=',')
print('csv file successfully generated')