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
	
	k_chr = 3
	k_wrd = 5

	y = [] 
	x_chr = []
	x_wrd = []

	max_word_len, max_sent_len, num_sent = 0, 0, 1000

	for line in lines[:num_sent]:
		words = line[:-1].split()
		tokens = words[1:]
		y.append(int(float(words[0])))
		max_sent_len = max(max_sent_len,len(tokens))
		for token in tokens:
			if token not in words_map:
				words_map[token] = word_cnt
				word_cnt += 1
				max_word_len = max(max_word_len,len(token))
			for i in xrange(len(token)):
				if token[i] not in char_map:
					char_map[token[i]] = char_cnt
					char_cnt += 1
	
	for line in lines[:num_sent]:
		words = line[:-1].split()
		tokens = words[1:]
		word_mat = [[0]*word_cnt]*(max_sent_len+k_wrd-1)
		char_mat = [[[0]*char_cnt]*(max_word_len+k_chr-1)]*(max_sent_len+k_wrd-1)

		for i in xrange(len(tokens)):
			word_mat[(k_wrd/2)+i][words_map[tokens[i]]]
			for j in xrange(len(tokens[i])):
				char_mat[(k_wrd/2)+i][(k_chr/2)+j][char_map[tokens[i][j]]]
		x_chr.append(char_mat)
		x_wrd.append(word_mat)
	max_word_len += k_chr-1
	max_sent_len += k_wrd-1
	# print char_cnt, word_cnt, max_word_len, max_sent_len
	#print numpy.shape(numpy.array(x_chr)), numpy.shape(numpy.array(x_wrd)), numpy.shape(numpy.array(y))
	data = (num_sent,char_cnt, word_cnt, max_word_len, max_sent_len, x_chr, x_wrd, y)
	return data
	#pickle.dump(data,open("data_mlp.pkl","wb"))

#read("tweets_clean.txt") 

