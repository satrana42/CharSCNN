import numpy, scipy, theano
import theano.tensor as T
import cPickle as pickle

class charnn(object):
	
	def __init__(self, rng, d, s, k, cl, W, C, b):
		self.d = d
		self.s = s
		self.k = k
		self.cl = cl

		if W == None:
			self.W = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(d+s)), high = sqrt(6/(d+s)), size=(s,d)),dtype=theano.config.floatX),borrow=True)
		else: self.W = W

		if C == None:
			self.C = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(d*k+cl)), high = sqrt(6/(d*k+cl)), size=(d*k,cl)),dtype=theano.config.floatX),borrow=True)
		else self.C = C

		if b == None:
			self.b  = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(d+s)), high = sqrt(6/(1+cl)), size=(cl)),dtype=theano.config.floatX),borrow=True)
		else self.W = W		

	def embed(self, inp):
		# inp: character matrix of word
		# inp size: word_len * size 
		# returns: character embedding of word
		# ret size: (word_len * s) * (s * d)
		return T.dot(inp,self.W)

	def conv(self, inp, word_len):
		# inp: character matrix of word
		# inp size: word_len * size 
		# returns: convolutional output
		# size: cl
		k = self.k
		d = self.d
		cl = self.cl
		inp = self.embed(inp)
		inp = numpy.append(numpy.array([[0]*d]*k),inp)
		inp = numpy.append(inp,numpy.array([[0]*d]*k))
		
		conv = numpy.asarray([-100000000 for i in xrange(cl)])
		for i in xrange(word_len):
			z = inp[k+i-(k/2):k+i+(k-(k/2))]
			z = T.reshape(z,(k*d))
			val = T.dot(z,C)+b
			for j in xrange(cl):
				conv[i] = max(conv[i],val[i])

		return theano.shared(conv, borrow=True)

class sentnn(object):

	def __init__(self, rng, cd, cs, ck, cl, wd, ws, wk, wl, cW, wW, cC, wC, cb, wb):
		
		self.cd = cd #character embedding dimension
		self.cs = cs #character vocab size
		self.ck = ck #character window length
		self.cl = cl #char conv layer out size

		self.wd = wd
		self.ws = ws
		self.wk = wk
		self.wl = wl
		
		#char embedding matrix
		if cW == None:
			self.cW = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(cd+cs)), high = sqrt(6/(cd+cs)), size=(cs,cd)),dtype=theano.config.floatX),borrow=True)
		else: self.cW = cW

		#char conv matrix
		if cC == None:
			self.cC = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(cd*ck+cl)), high = sqrt(6/(cd*ck+cl)), size=(cd*ck,cl)),dtype=theano.config.floatX),borrow=True)
		else self.cC = cC

		#char conv bias
		if cb == None:
			self.cb  = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(cl+1)), high = sqrt(6/(cl+1)), size=(cl)),dtype=theano.config.floatX),borrow=True)
		else self.cb = cb	

		if wW == None:
			self.wW = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(wd+ws)), high = sqrt(6/(wd+ws)), size=(ws,wd)),dtype=theano.config.floatX),borrow=True)
		else: self.wW = wW

		if wC == None:
			self.wC = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(wd*wk+wl)), high = sqrt(6/(wd*wk+wl)), size=(wd*wk,wl)),dtype=theano.config.floatX),borrow=True)
		else self.wC = wC

		if wb == None:
			self.wb  = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(wl+1)), high = sqrt(6/(1+wl)), size=(wl)),dtype=theano.config.floatX),borrow=True)
		else self.wb = wb

		ch = charnn(rng=rng, d=cd, s=cs, k=ck, cl=cl, W=cW, C=cC, b=cb)
		self.params = [cW, wW, cC, wC, cb, wb]

	def embed(self, inp, sent_len, sentvec):
		# inp: word matrix of sentence
		# inp size: sent_len * word_vocab_size 
		# returns: word + char embedding of sentence
		# size: sent_len * (cd+wd)
		wemb = T.dot(inp, self.wW)

		for i in xrange(sent_len):
			cemb = ch.conv( w2c(sentvec[i]), wlen(sentvec[i]) )
			numpy.append(wemb[i],cemp)

		return theano.shared(wemb, borrow=True)

	def conv(self, inp, sent_len, sentvec):
		# inp: word matrix of sent
		# inp size: sent_len * word_vocab_size 
		# sent_len: length of sentence
		# sentvec: list containing indices of words
		# returns: convolutional output
		# size: wl
		k = self.wk
		d = self.wd
		wl = self.wl
		inp = self.embed(inp,sent_len,sentvec)
		inp = numpy.append(numpy.array([[0]*d]*k),inp)
		inp = numpy.append(inp,numpy.array([[0]*d]*k))
		conv = numpy.asarray([-100000000]*wl)
		for i in xrange(sent_len):
			z = inp[k+i-(k/2):k+i+(k-(k/2))]
			z = T.reshape(z,(k*d))
			val = T.dot(z,C)+b
			for j in xrange(wl):
				conv[i] = max(conv[i],val[i])
		return theano.shared(conv, borrow=True)

	def eval(self, inp, sent_len, sentvec):
		return T.nnet.sigmoid(self.conv(inp,sent_len,sentvec))

def train(learning_rate=0.1, training_epochs=10, char_vocab_size, word_vocab_size, sentvec, inp):
	#updates neural net with a sentence as input
	charscnn = sentnn(rng=numpy.random.RandomState(123), cd=5, cs=char_vocab_size, ck=3, cl=50, wd=30, ws=word_vocab_size, wk=5, wl=300)
	
	idx  =T.scalar('idx')
	x = T.matrix('x')
	s = T.vector('s')
	l = T.scalar('l')
	y = T.vector('y')
	
	lr = LogisticRegression(input=charscnn.eval(x, l, s), n_in=300, n_out=1)
	
	# the cost we minimize during training is the NLL of the model
	cost = lr.negative_log_likelihood(y)
	
	params = self.params+lr.params
	grads = T.grad(cost, params)
	
	updates = []
	for param_i, grad_i in zip(params, grads):
		updates.append((param_i, param_i - learning_rate * grad_i))

	train_cnn = theano.function([idx], [T.mean(cost)], updates=updates, givens={x:inp[idx], s:sentvec[idx], l:len(sentvec[idx])}, mode="FAST_RUN")

    start_time = time.clock()
    
    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for i in xrange(len(sentvec)):
            c.append(train_cnn(i))
        c_array = numpy.vstack(c)
        print 'Training epoch %d, reconstruction cost ' % epoch, numpy.mean(c_array[0]), ' jacobian norm ', numpy.mean(numpy.sqrt(c_array[1]))
    end_time = time.clock()

    training_time = (end_time - start_time)/60.

    print "Training Time: ", training_time
	

#data loading code here
char_vocab_size, word_vocab_size, sentvec, inp, word_char_mat = pickle.load("data.pkl","rb")

def w2c(idx):
	#returns character matrix of a word #idx
	return word_char_mat[idx]

def wlen(idx):
	#returns len of word #idx
	return len(word_char_mat[idx])

train(char_vocab_size=char_vocab_size, word_vocab_size=word_vocab_size, sentvec=sentvec, inp=inp)



