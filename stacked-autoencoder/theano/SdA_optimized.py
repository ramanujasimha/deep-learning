"""
 Code based on SdA by Vincent et al: 
 http://deeplearning.net/tutorial/SdA.html

 Code optmized (using suggested tricks) by: Ramanuja Simha
"""

from __future__ import print_function

import six.moves.cPickle as pickle
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA


# start-snippet-1
class SdAOptimized(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        W_layers,
        b_layers,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type W_layers: list
        :param W_layers: Weights learned for denoising autoencoder layers; 
                          each value is of type theano.tensor.TensorType

        :type b_layers: list
        :param b_layers: Bias values for denoising autoencoder layers;  
                          each value is of type theano.tensor.TensorType

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        """

        self.sigmoid_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        W=W_layers[i],
                                        b=b_layers[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)
        # end-snippet-2
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score

def run_dA(numpy_rng, train_set_x, size, layer=0, W=None, bhid=None,
           learning_rate=0.1, training_epochs=15, batch_size=1,
           n_visible=28*28, n_hidden=500, corruption_level=0.1, 
           train_mode=1
    ):
    # compute number of minibatches for training, validation and testing
    n_train_batches = \
          train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    ######################
    # BUILDING THE MODEL #
    ######################

    da = None
    if train_mode:
        da = dA(
            numpy_rng=numpy_rng,
            input=x,
            n_visible=n_visible,
            n_hidden=n_hidden
        )
    else:
        da = dA(
            numpy_rng=numpy_rng,
            input=x,
            n_visible=n_visible,
            n_hidden=n_hidden,
            W=W,
            bhid=bhid
        )

    cost, updates = da.get_cost_updates(
        corruption_level=corruption_level,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    hidden_vals = da.get_hidden_values(
        input=x
    )

    get_hidden_da = theano.function(
        [index],
        hidden_vals,
        givens={
            x: train_set_x[index * size: (index + 1) * size]
        }
    )

    corrupted_in = da.get_corrupted_input(
        input=x,
        corruption_level=0.1
    )

    get_corrupted_da = theano.function(
        [index],
        corrupted_in,
        givens={
            x: train_set_x[index * size: (index + 1) * size]
        }
    )

    ############
    # TRAINING #
    ############

    if train_mode:
        start_time = timeit.default_timer()

        # go through training epochs
        for epoch in range(training_epochs):
            # go through trainng set
            c = []
            for batch_index in range(n_train_batches):
                c.append(train_da(batch_index))

            print('Pre-training layer %i, epoch %d, cost %f' %
                (layer, epoch, numpy.mean(c, dtype='float64')))

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print(('The pre-training code for layer %i ran for %.2fm' %
               (layer, (training_time) / 60.)), file=sys.stderr)

    start_time = timeit.default_timer()

    #y = numpy.ndarray((0, n_hidden), float)
    #for batch_index in range(n_train_batches):
    #    y = numpy.append(y, get_hidden_da(batch_index), 0)
    y = get_hidden_da(0)

    """
    # get data with noise
    tilde_x = numpy.ndarray((0, n_visible), float)
    for batch_index in range(n_train_batches):
        tilde_x = numpy.append(tilde_x, get_corrupted_da(batch_index), 0)
    """

    end_time = timeit.default_timer()

    fetching_hdata_time = (end_time - start_time)

    print(('The fetching hidden data code ' +
           'ran for %.2fm' % ((fetching_hdata_time) / 60.)), 
           file=sys.stderr)

    if train_mode:
        return da.W, da.b, y   #,  tilde_x
    else:
        return y

def test_SdAOptimized(finetune_lr=0.1, pretraining_epochs=15,
            	      pretrain_lr=0.001, training_epochs=1000,
             	      dataset='../../data/mnist.pkl.gz', batch_size=1,
                      n_ins=28*28, n_outs=10,
                      hidden_layers_sizes=[500, 500, 500],
                      corruption_levels = [.1, .2, .3]
    ):

    datasets = load_data(dataset)
    data, size = datasets[0], datasets[1]

    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x, test_set_y = data[2]

    train_size, valid_size, test_size = size[0], size[1], size[2]

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)

    #if not os.path.isdir(output_folder):
    #    os.makedirs(output_folder)
    #os.chdir(output_folder)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... pre-training the model')
    start_time = timeit.default_timer()

    n_layers = len(hidden_layers_sizes)
    W_layers, b_layers = [], []
    y_layer = numpy.ndarray

    ## Pre-train layer-wise
    for i in range(n_layers):
        if i ==0:
            W, b, y = run_dA(numpy_rng=numpy_rng,
                             train_set_x=train_set_x, 
                             size=train_size, layer=i,
                             learning_rate=pretrain_lr,
                             training_epochs=pretraining_epochs,
                             n_visible=n_ins,
                             n_hidden=hidden_layers_sizes[i],
                             corruption_level=corruption_levels[i],
                             train_mode=1)
            W_layers.append(W)
            b_layers.append(b)
            y_layer = y
        else:
            shared_y_layer = \
                    theano.shared(numpy.asarray(
                                        y_layer, dtype=theano.config.floatX),
                                  borrow=True)
            W, b, y = run_dA(numpy_rng=numpy_rng,
                             train_set_x=shared_y_layer, 
                             size=train_size, layer=i,
                             learning_rate=pretrain_lr,
                             training_epochs=pretraining_epochs,
                             n_visible=hidden_layers_sizes[i-1],
                             n_hidden=hidden_layers_sizes[i],
                             corruption_level=corruption_levels[i],
                             train_mode=1)
            W_layers.append(W)
            b_layers.append(b)
            y_layer = y

    end_time = timeit.default_timer()

    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), 
           file=sys.stderr)
    # end-snippet-4

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # start-snippet-3
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdAOptimized(
        numpy_rng=numpy_rng,
        W_layers=W_layers,
        b_layers=b_layers,
        n_ins=n_ins,
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=n_outs
    )
    # end-snippet-3 start-snippet-4

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=data,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = \
                            numpy.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    # get transformations for train, valid, and test datasets
    y_train, y_valid, y_test = numpy.ndarray, numpy.ndarray, numpy.ndarray
    for i in range(n_layers):
        W_final = sda.sigmoid_layers[i].W
        b_final = sda.sigmoid_layers[i].b
        if i ==0:
            y_train = run_dA(numpy_rng=numpy_rng, train_set_x=train_set_x, size=train_size,
                             layer=i, W=W_final, bhid=b_final, n_visible=n_ins,
                             n_hidden=hidden_layers_sizes[i],
                             corruption_level=corruption_levels[i],
                             train_mode=0)
            y_valid = run_dA(numpy_rng=numpy_rng, train_set_x=valid_set_x, size=valid_size,
                             layer=i, W=W_final, bhid=b_final, n_visible=n_ins,
                             n_hidden=hidden_layers_sizes[i],
                             corruption_level=corruption_levels[i],
                             train_mode=0)
            y_test = run_dA(numpy_rng=numpy_rng, train_set_x=test_set_x, size=test_size,
                            layer=i, W=W_final, bhid=b_final, n_visible=n_ins,
                            n_hidden=hidden_layers_sizes[i],
                            corruption_level=corruption_levels[i],
                            train_mode=0)
        else:
            shared_y_train = \
                    theano.shared(numpy.asarray(
                                        y_train, dtype=theano.config.floatX),
                                  borrow=True)
            y_train = run_dA(numpy_rng=numpy_rng, train_set_x=shared_y_train, size=train_size,
                             layer=i, W=W_final, bhid=b_final,
                             n_visible=hidden_layers_sizes[i-1],
                             n_hidden=hidden_layers_sizes[i],
                             corruption_level=corruption_levels[i],
                             train_mode=0)
            shared_y_valid = \
                    theano.shared(numpy.asarray(
                                        y_valid, dtype=theano.config.floatX),
                                  borrow=True)
            y_valid = run_dA(numpy_rng=numpy_rng, train_set_x=shared_y_valid, size=valid_size,
                             layer=i, W=W_final, bhid=b_final,
                             n_visible=hidden_layers_sizes[i-1],
                             n_hidden=hidden_layers_sizes[i],
                             corruption_level=corruption_levels[i],
                             train_mode=0)
            shared_y_test = \
                    theano.shared(numpy.asarray(
                                        y_test, dtype=theano.config.floatX),
                                  borrow=True)
            y_test = run_dA(numpy_rng=numpy_rng, train_set_x=shared_y_test, size=test_size,
                            layer=i, W=W_final, bhid=b_final,
                            n_visible=hidden_layers_sizes[i-1],
                            n_hidden=hidden_layers_sizes[i],
                            corruption_level=corruption_levels[i],
                            train_mode=0)

    print(y_train.shape, y_valid.shape, y_test.shape)

if __name__ == '__main__':
    test_SdAOptimized()
