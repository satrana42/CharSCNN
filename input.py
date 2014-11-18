from twokenize import tokenizeRawTweetText as tokenize
from scipy.sparse import *
from scipy import *
from scipy import linalg, mat, dot
from sklearn import *
import numpy as np
import cPickle as pickle
import random

def read(inp_file):
	f_in = open(inp_file, 'r')
	lines = f_in.readlines()
	words_map = {}

	cnt = 0
	scores = []
	tokenized_tweets = []

	for line in lines:
		words = line[:-1].split()
		tokens = tokenize(' '.join(words[3:]))
		tokenized_tweets.append(tokens)
		scores.append(float(words[2]))
		for token in tokens:
			if token not in words_map:
				words_map[token]=cnt
				cnt += 1

	print 'tokenized'

	x = dok_matrix((len(scores), len(words_map)), dtype=float64)
	y = numpy.asarray(scores, dtype=float64)

	i=0
	for tokens in tokenized_tweets:
		for token in tokens:
			x[i,words_map[token]]=1
		i+=1
	print 'matrix created'


