"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import load

import numpy
import cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.1, n_epochs=10,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    d_wrd = 10
    k_wrd = 5
    d_char = 5
    k_char = 3
    cl_char = 10
    cl_wrd = 50

    rng = numpy.random.RandomState(23455)

    print "loading"
    #(num_sent, v_char, v_wrd, max_word_len, max_sen_len, set_char, set_wrd, set_y) = pickle.load(open("data_mlp.pkl","rb"))
    (num_sent, v_char, v_wrd, max_word_len, max_sen_len, set_char, set_wrd, set_y) = load.read("tweets_clean.txt")
    print "loaded"
    set_char = theano.shared(numpy.array(set_char,dtype=theano.config.floatX),borrow=True)
    set_wrd = theano.shared(numpy.array(set_wrd,dtype=theano.config.floatX),borrow=True)
    set_y = theano.shared(numpy.array(set_y),borrow=True)
    print "prepared"
    n_train_batches = 8*num_sent/10
    n_valid_batches = num_sent/10
    n_test_batches = num_sent/10
    
    train_x_wrd, train_x_char, train_y = set_wrd[:n_train_batches], set_char[:n_train_batches], set_y[:n_train_batches]
    val_x_wrd, val_x_char, val_y = set_wrd[n_train_batches:n_train_batches+n_valid_batches], set_char[n_train_batches:n_train_batches+n_valid_batches], set_y[n_train_batches:n_train_batches+n_valid_batches]
    test_x_wrd, test_x_char, test_y = set_wrd[-n_test_batches:], set_char[-n_test_batches:], set_y[-n_test_batches:]

    # compute number of minibatches for training, validation and testing
    
    
    #theano.config.compute_test_value = 'warn'
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x_wrd = T.matrix('x_wrd')   # the data is presented as rasterized images
    x_char = T.tensor3('x_char') 
    y = T.lvector('y')

    # x_char.tag.test_value = numpy.random.rand(max_sen_len,max_word_len,v_char)
    # x_wrd.tag.test_value = numpy.random.rand(max_sen_len,v_wrd)
    # y.tag.test_value = numpy.array([1])

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x_char.reshape((max_sen_len, 1, max_word_len, v_char))
    
    layer0 = HiddenLayer(
        rng,
        input=layer0_input,
        n_in=v_char,
        n_out=d_char,
        isb=0
    )
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(max_sen_len, 1, max_word_len, d_char),
        filter_shape=(cl_char, 1, k_char, d_char),
        poolsize=(max_word_len - k_char + 1, 1)
    )

    layer2_input = x_wrd.reshape((max_sen_len, v_wrd))
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=v_wrd,
        n_out=d_wrd,
        isb=0
    )
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)
    layer3_input = T.concatenate([layer1.output.reshape((max_sen_len,cl_char)), layer2.output], axis=1).reshape((1, 1, max_sen_len, cl_char + d_wrd))

    layer3 = LeNetConvPoolLayer(
        rng,
        input=layer3_input,
        image_shape=(1, 1, max_sen_len, cl_char + d_wrd),
        filter_shape=(cl_wrd, 1, k_wrd, cl_char + d_wrd),
        poolsize=(max_sen_len - k_wrd + 1, 1)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer4_input = layer3.output.reshape((1,cl_wrd))

    # construct a fully-connected sigmoidal layer
    layer4 = HiddenLayer(
        rng,
        input=layer4_input,
        n_in=cl_wrd,
        n_out=50,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer5 = LogisticRegression(input=layer4.output, n_in=50, n_out=2)

    # the cost we minimize during training is the NLL of the model
    #theano.printing.Print('this is a very important value')(x_chr)
    cost = layer5.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer5.errors(y),
        givens={
            x_wrd: test_x_wrd[index],
            x_char: test_x_char[index],
            y: test_y[index:index+1]
        },
        mode="FAST_RUN"
    )

    validate_model = theano.function(
        [index],
        #layer5.negative_log_likelihood(y),
        layer5.errors(y),
        givens={
            x_wrd: val_x_wrd[index],
            x_char: val_x_char[index],
            y: val_y[index:index+1]
        },
        mode="FAST_RUN"
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x_wrd: train_x_wrd[index],
            x_char: train_x_char[index],
            y: train_y[index:index+1]
        },
        mode="FAST_RUN"
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            # ((theano.printing.Print(x)))
            cost_ij =  train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


print "start"
evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
