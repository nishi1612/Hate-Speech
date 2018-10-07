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

'''Features will be readability scores, sentiments, no of words

Extra features to try no of syllables, no of characters
For improvement of model, we can use semantic features and remove hashtags properly
Also, later try LSTM, RNN
Now use Decision Tree

'''


def break_sentences(text):
	nlp = spacy.load('en')
	doc = nlp(text)
	return doc.sents

# Returns Number of Words in the text
def word_count(text):
	sentences = break_sentences(text)
	words = 0
	for sentence in sentences:
		words += len([token for token in sentence])
	return words

# Returns the number of sentences in the text
def sentence_count(text):
	sentences = break_sentences(text)
	return len(sentences)

# Returns average sentence length
def avg_sentence_length(self, text):
	words = word_count(text)
	sentences = sentence_count(text)
	average_sentence_length = float(words / sentences)
	return average_sentence_length

# Textstat is a python package, to calculate statistics from
# text to determine readability,
# complexity and grade level of a particular corpus.
# Package can be found at https://pypi.python.org/pypi/textstat
def syllables_count(word):
	return textstatistics().syllable_count(word)

# Returns the average number of syllables per
# word in the text
def avg_syllables_per_word(text):
	syllable = syllables_count(text)
	words = word_count(text)
	ASPW = float(syllable) / float(words)
	return legacy_round(ASPW, 1)

# Return total Difficult Words in a text
def difficult_words(text):

    # Find all words in the text
	words = []
	#sentences = break_sentences(text)
	for sentence in sentences:
		words += [token for token in sentence]

	# difficult words are those with syllables >= 2
	# easy_word_set is provide by Textstat as
	# a list of common words
	diff_words_set = set()

	for word in words:
		syllable_count = syllables_count(word)
		if word not in easy_word_set and syllable_count >= 2:
			diff_words_set.add(word)

	return len(diff_words_set)
