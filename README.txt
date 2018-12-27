## NECESSARY LIBRARIES
## import data processing/cleaning , data modeling libraries

import pandas as pd
import os
import sys
import re as re
import datetime as datetime
import numpy as np
import collections
import string
import pandas as pd

from nltk.tag.perceptron import PerceptronTagger
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords

from gensim.models import Doc2Vec
#from gensim.models import Word2Vec
from gensim.models import fasttext

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from gensim.models.wrappers import FastText
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings

import tensorflow as tf
import tensorflow_hub as hub

## FACEBOOK PRE-TRAINED MODEL
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md (chose english and unzip the file)