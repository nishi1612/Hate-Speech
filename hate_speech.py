import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
import html

hate_speech_data = pd.read_csv('labeled_data.csv')
tweet = []
count = 0

for t in hate_speech_data['tweet']:
	t = html.unescape(t)
	z = lambda x: re.compile('\#').sub('', re.compile('RT @').sub('@', x).strip())
	t = z(t)
	tweet.append(' '.join(re.sub("(@[_A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",t).split()))



