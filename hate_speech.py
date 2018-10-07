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


# A word is polysyllablic if it has more than 3 syllables
# this functions returns the number of all such words
# present in the text
def poly_syllable_count(text):
	count = 0
	words = []
	sentences = break_sentences(text)
	for sentence in sentences:
		words += [token for token in sentence]


	for word in words:
		syllable_count = syllables_count(word)
		if syllable_count >= 3:
			count += 1
	return count


def flesch_reading_ease(text):
	"""
	Implements Flesch Formula:
	Reading Ease score = 206.835 - (1.015 × ASL) - (84.6 × ASW)
	Here,
		ASL = average sentence length (number of words
	        divided by number of sentences)
		ASW = average word length in syllables (number of syllables
	        divided by number of words)
	"""
	FRE = 206.835 - float(1.015 * avg_sentence_length(text)) - float(84.6 * avg_syllables_per_word(text))
	return legacy_round(FRE, 2)


def gunning_fog(text):
	per_diff_words = (difficult_words(text) / word_count(text) * 100) + 5
	grade = 0.4 * (avg_sentence_length(text) + per_diff_words)
	return grade


def smog_index(text):
	"""
        Implements SMOG Formula / Grading
        SMOG grading = 3 + ?polysyllable count.
        Here,
           polysyllable count = number of words of more
          than two syllables in a sample of 30 sentences.
	"""
	if sentence_count(text) >= 3:
		poly_syllab = poly_syllable_count(text)
		SMOG = (1.043 * (30*(poly_syllab / sentence_count(text)))**0.5) + 3.1291
		return legacy_round(SMOG, 1)
	else:
		return 0

def dale_chall_readability_score(self, text):
	"""
        Implements Dale Challe Formula:
        Raw score = 0.1579*(PDW) + 0.0496*(ASL) + 3.6365
        Here,
            PDW = Percentage of difficult words.
            ASL = Average sentence length
	"""
	words = word_count(text)
    # Number of words not termed as difficult words
	count = word_count - difficult_words(text)
	if words > 0:
        # Percentage of words not on difficult word list
		per = float(count) / float(words) * 100
    # diff_words stores percentage of difficult words
	diff_words = 100 - per
	raw_score = (0.1579 * diff_words) + (0.0496 * avg_sentence_length(text))

    # If Percentage of Difficult Words is greater than 5 %, then;
    # Adjusted Score = Raw Score + 3.6365,
    # otherwise Adjusted Score = Raw Score

	if diff_words > 5:
		raw_score += 3.6365
	return legacy_round(score, 2)

def getFeatureArray(tweets):
	features = []
	sentiment_analyzer = vader()
	for t in tweets:
		t = preprocess(t).lower()
		FRE = flesch_reading_ease(t)
		no_of_difficult_words = difficult_words(t)
		dale_chall_readability = dale_chall_readability_score(t)
		t = tokenizer(t)
		sentiment = sentiment_analyzer.polarity_scores(t)
		no_of_words = len(t)
		features.append(sentiment, FRE, dale_chall_readability, no_of_words, no_of_difficult_words)
