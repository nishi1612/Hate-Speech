import warnings
warnings.simplefilter(action = 'ignore', category = DeprecationWarning)

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import html
import nltk
import nltk.sentiment.vader as vader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *

hate_speech_data = pd.read_csv('labeled_data.csv')

stopwords = nltk.corpus.stopwords.words("english")
stopwords.extend('rt')

lemmatizer = WordNetLemmatizer()
def preprocess(tweets):
	t = html.unescape(tweets)
	z = lambda x: re.compile('\#').sub('', re.compile('RT @').sub('@', x).strip())
	t = z(t)
	tweet = ' '.join(re.sub("(@[_A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",t).split())
	return tweet


def tokenizer(tweets):
	tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweets.lower())).strip()
	tweets = [lemmatizer.lemmatize(tweet) for tweet in tweets.split()]
	return tweets



vectorizer = TfidfVectorizer(
	tokenizer = tokenizer,
	preprocessor = preprocess,
	decode_error = 'replace',
	stop_words = stopwords,
	ngram_range = (1,3),
	max_features=10000,
	min_df=5,
	max_df=0.75
	)

tweet = vectorizer.fit_transform(hate_speech_data['tweet']).toarray()

def pos(tokenized_tweet):
	return nltk.pos_tag(tokenized_tweet)

def pos_tweet(tweet):
	total_pos_tags = []
	count = 0
	for t in tweet:
		t = preprocess(t)
		t = tokenizer(t)
		pos_dict = pos(t)
		pos_tags = pos_dict[1]
		pos_tags = " ".join(pos_tags)

		total_pos_tags.append(pos_tags)
		print(total_pos_tags)

pos_tweet(hate_speech_data['tweet'])