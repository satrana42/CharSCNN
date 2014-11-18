import numpy, scipy, theano
import theano.tensor as T

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
		# inp: word_len * size 
		# returns character embedding of word
		# size: (word_len * s) * (s * d)
		return T.dot(inp,self.W)

	def conv(self, inp, word_len):
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
		# returns convolutional output
		# size: cl
		return theano.shared(conv, borrow=True)

class sentnn(object):

	def __init__(self, rng, inp, batch_size, cd, cs, ck, cl, wd, ws, wk, wl, cW, wW, cC, wC, cb, wb):

		# self.batch_size = batch_size
		# self.inp = inp
		
		self.cd = cd
		self.cs = cs
		self.ck = ck
		
		self.wd = wd
		self.ws = ws
		self.wk = wk
		self.wl = wl
		
		if cW == None:
			self.cW = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(cd+cs)), high = sqrt(6/(cd+cs)), size=(cs,cd)),dtype=theano.config.floatX),borrow=True)
		else: self.cW = cW

		if cC == None:
			self.cC = theano.shared(numpy.asarray(rng.uniform(low = -sqrt(6/(cd*ck+cl)), high = sqrt(6/(cd*ck+cl)), size=(cd*ck,cl)),dtype=theano.config.floatX),borrow=True)
		else self.cC = cC

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
		# inp: sent_len *  s
		# returns word + char embedding of sentence
		# size: sent_len * (cd+wd)
		wemb = T.dot(inp, self.wW)

		for i in xrange(sent_len):
			cemb = ch.conv( w2c(sentvec[i]), wlen(sentvec[i]) )
			numpy.append(wemb[i],cemp)

		return theano.shared(wemb, borrow=True)

	def conv(self, inp, sent_len, sentvec):
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
		# returns convolutional output
		# size: wl
		return theano.shared(conv, borrow=True)

	def eval(self, inp, sent_len, sentvec):
		return T.nnet.sigmoid(self.conv(inp,sent_len,sentvec))

	def update(self, learning_rate, inp, sent_len, sentvec):
		prob = self.eval(inp,sent_len,sentvec)
		params = self.params
		grads = T.grad(cost, params)
		updates = []
    	for param_i, grad_i in zip(params, grads):
        	updates.append((param_i, param_i - learning_rate * grad_i))
		# Do stochastic gradient descent

def train(learning_rate,):




