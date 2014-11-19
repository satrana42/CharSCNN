from twokenize import tokenizeRawTweetText as tokenize
import numpy, theano
import theano.tensor as T
import cPickle as pickle

def read(inp_file):
	f_in = open(inp_file, 'r')
	lines = f_in.readlines()
	words_map = {}
	char_map = {}
	word_cnt = 0
	char_cnt = 0
	
	scores = []
	tokenized_tweets = []
	sentvec=[]
	
	for line in lines[:100]:
		words = line[:-1].split()
		tokens = tokenize(' '.join(words[3:]))
		tokenized_tweets.append(tokens)
		scores.append(int(float(words[2])))
		sent = []
		for token in tokens:
			if token not in words_map:
				words_map[token] = word_cnt
				word_cnt += 1
			for i in xrange(len(token)):
				if token[i] not in char_map:
					char_map[token[i]] = char_cnt
					char_cnt += 1
			sent.append(words_map[token])
		sentvec.append(sent)

	inp, word_char_mat = [], []
	
	for (word,idx) in words_map.iteritems():
		charmat = [[0]*char_cnt]*len(word)
		for i in xrange(len(word)):
			charmat[i][char_map[word[i]]]=1
		word_char_mat.append(charmat)

	print inp, word_char_mat

	for sent in tokenized_tweets:
		wordmat = [[0]*word_cnt]*len(sent)
		for i in xrange(len(sent)):
			wordmat[i][words_map[sent[i]]]=1
		inp.append(charmat)
	
	data = (char_cnt, word_cnt, sentvec, inp, scores, word_char_mat)
	pickle.dump(data,open("data.pkl","wb"))

read("tweets.txt") 

