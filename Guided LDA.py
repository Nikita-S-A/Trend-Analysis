# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:09:20 2024

@author: mail2
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import pandas as pd
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.DEBUG)

# Ensure you have the dataset 'input.csv' in the appropriate directory
df = pd.read_csv('input.csv')
# Assuming 'text' column holds your textual data
texts = df['MsgBody'].fillna('').tolist()

# Tokenization and stopwords removal
tokenizer = RegexpTokenizer(r'\w+')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercasing and tokenization
    tokens = tokenizer.tokenize(text.lower())
    # Removing stopwords
    tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
    return tokens

# Apply preprocessing
processed_docs = [preprocess(text) for text in texts]

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Create a bag-of-words corpus
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Set training parameters
num_topics = 5
passes = 10
model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes)

# Print the topics
topics = model.print_topics(num_words=4)
for topic in topics:
    print(topic)
    
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Prepare visualization
pyLDAvis.enable_notebook()
vis = gensimvis.prepare(model, corpus, dictionary)
pyLDAvis.show(vis)

model.save('lda_model.gensim')
