""" 
 Author: Ramanuja Simha - https://github.com/ramanujasimha
 License: MIT License

 This piece of code attempts to implement a stacked autoencoder using 
 constructs provided by a high-level neural network library named Keras.
 NOTE: Exception handling has yet to be implemented extensively. 
"""

import six.moves.cPickle as pickle
import gzip

import sys
import numpy

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

class StackedAutoencoder(object):
    """Stacked auto encoder class (StackedAutoEncoder)

      An autoencoder tries to compute a latent representation of the 
      input. A stacked autoencoder finds multiple such latent 
      representations, where the final representation most accurately 
      distinguishes the classes. To enable learning of a representation 
      that can separate classes, the final layer is set as softmax. 
    """
      
    def __init__(
        self,
        numpy_rs,
        n_visible,
        n_predicted,
        n_hidden_layers,
        dropouts,
        hidden_layers,
        activation='sigmoid',
        final_activation='softmax'
    ):
        """
        Initialize the StackedAutoencoder class by specifying the number 
        of visible units (i.e. input dimension), the number of units in 
        in each hidden layer, the dropouts associated with each layer, 
        and the number of predicted units (i.e. output dimension). 
        The constructor also takes as input a numpy random seed - 
        which is set for reproducibility - and activation functions 
        for each hidden layer as well as the final layer. 

        :type numpy_rs: integer (required) 
        :param numpy_rs: used to generate numpy random seed

        :type n_visible: integer (required)
        :param n_visible: number of visible units

        :type n_predicted: integer (required)
        :param n_predicted: number of predicted units

        :type n_hidden_layers: integer (required)
        :param n_hidden_layers: number of hidden layers

        :type dropouts: list (required)
        :param dropouts: dropout value for each hidden layer

        :type hidden_layers: list (required)
        :param hidden_layers: number of units in each hidden layer

        :type activation: function name (optional)
        :param activation: activation function used in each hidden layer
                          set to 'sigmoid' by default
        :type final_activation: function name (optional)
        :param final_activation: activation function used in the final layer
                          set to 'softwax" by default
        """
        self.numpy_rs = numpy_rs
        self.n_visible = n_visible
        self.n_predicted = n_predicted
        self.n_hidden_layers = n_hidden_layers
        self.hidden_layers = hidden_layers
        self.activation = activation

        # fix random seed for reproducibility
        numpy.random.seed(self.numpy_rs)

        # set-up model
        print "SETTING UP A STACKED AUTO-ENCODER ..."
        self.model = Sequential()
        self.model.add(Dense(input_dim=n_visible, output_dim=hidden_layers[0],
                             activation=activation))
        self.model.add(Dropout(dropouts[0]))
        print "Adding hidden layer .. in_dim:",n_visible, \
                                            "out_dim:",hidden_layers[0]
        print "Adding dropout layer ..",dropouts[0]
        for j in range(1, n_hidden_layers):
            self.model.add(Dense(input_dim=hidden_layers[j-1],
                                 output_dim=hidden_layers[j],
                                 activation=activation))
            self.model.add(Dropout(dropouts[j]))
            print "Adding hidden layer .. in_dim:",hidden_layers[j-1], \
                                            "out_dim:",hidden_layers[j]
            print "Adding dropout layer ..",dropouts[j]
        self.model.add(Dense(input_dim=hidden_layers[n_hidden_layers-1],
                             output_dim=n_predicted,
                             activation=final_activation))
        print "Adding output layer .. in_dim:", \
                    hidden_layers[n_hidden_layers-1],"out_dim:",n_predicted

    def train(self,
              X_train,
              y_train,
              X_valid,
              y_valid,
              class_weights=None,
              loss='categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy','fmeasure'],
              n_epoch=25,
              batch_size=16
    ):
        """This function trains the autoencoder.

        type X_train: numpy 2-d array (required)
        param X_train: feature vectors for instances used for model training

        type y_train: numpy 1-d array (required)
        param y_train: class values for instances used for model training

        type X_valid: numpy 2-d array (required)
        param X_valid: feature vectors for instnaces used for model validation

        type y_valid: numpy 1-d array (required)
        param y_valid: class values for instances used for model validation

        type class_weights: dictionary (optional)
        param class_weights: weight for each class used to train network
                      - particularly used in case of class imbalance
                      set to 'None' by default

        type loss: function name (optional)
        param loss: loss function used to train network
                      set to 'categorical_crossentropy' by default

        type optimizer: function name (optional)
        param optimizer: optimization method used to train network
                      set to 'adam' by default

        type metrics: list (optional)
        param metrics: measures used to evaluate predictions on validation data
                      set to 'accuracy' and 'fmeasure' by default

        type n_epoch: integer (optional)
        param n_epoch: number of iterations to train network
                      set to 25 by default

        type batch_size: integer (optional)
        param batch_size: number of instances used to optmize model parameters
                      set to 16 by default
        """
        # transform representation from binary to categorical
        Y_train = to_categorical(y_train, nb_classes=None)
        Y_valid = to_categorical(y_valid, nb_classes=None)

        # train model
        print "Training stacked denoising autoencoder ..."
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if class_weights is None:
            self.model.fit(x=X_train, y=Y_train,
                           validation_data=tuple((X_valid,Y_valid)),
                           nb_epoch=n_epoch, batch_size=batch_size)
        else:
            self.model.fit(x=X_train, y=Y_train, class_weight=class_weights,
                           validation_data=tuple((X_valid,Y_valid)),
                           nb_epoch=n_epoch, batch_size=batch_size)

    def get_hidden_representation(self, X_train, X_valid, X_test):
        """This function gets hidden representation of input.
        """
        # build new model with activations of old model
        # truncate model after hidden layer
        model2 = Sequential()
        model2.add( \
                Dense(input_dim=self.n_visible,
                      output_dim=self.hidden_layers[0],
                      activation=self.activation,
                      weights=self.model.layers[0].get_weights()))
        for j in range(1, self.n_hidden_layers):
            model2.add(Dense(input_dim=self.hidden_layers[j-1],
                             output_dim=self.hidden_layers[j],
                             activation=self.activation,
                             weights=self.model.layers[j*2].get_weights()))

        # predict hidden layer values
        print "Retrieving hidden representation ..."
        X_train_enc = model2.predict(X_train, batch_size=X_train.shape[0])
        X_valid_enc = model2.predict(X_valid, batch_size=X_valid.shape[0])
        X_test_enc = model2.predict(X_test, batch_size=X_test.shape[0])

        return X_train_enc, X_valid_enc, X_test_enc

    def predict(self, X):
        """This function obtains class predictions for input features. 
        """
        return self.model.predict(X).argmax(1)

def test_stacked_autoencoder(activation='sigmoid', 
                             final_activation='softmax', 
                             loss='categorical_crossentropy',
                             optimizer='adam', 
                             metrics = ['accuracy','fmeasure'],
                             training_epochs=5, 
                             batch_size=16, 
                             input_file='data/mnist.pkl.gz'
):
    """This function is used to test working of class StackedAutoencoder. 
    
    Once hidden representation of instances is obtained from stacked autoencoder, 
    this set of features is fed to a Random Foreset classifier. For fun, accuracy
    and f-1 score of class predictions directly obtained from the network are 
    compared with that of predictions obtained using the Random Forest classifier. 
    """
    # load well-known MNIST dataset available at:
    # 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    train_set, valid_set, test_set = numpy.array, numpy.array, numpy.array
    with gzip.open(input_file, 'rb') as f: 
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1') 
        except:
            train_set, valid_set, test_set = pickle.load(f)

    X_train, y_train = train_set[0], train_set[1]
    X_valid, y_valid = valid_set[0], valid_set[1]
    X_test, y_test = test_set[0], test_set[1]

    # set input variables
    numpy_seed = 7
    n_visible = X_train.shape[1]
    n_predicted = len(set(y_train))
    n_hidden_layers = 2
    dropouts = [.2, .2]
    hidden_layers = [int(n_visible/10.), int(n_visible/10.)]

    # initialize stacked autoencoder
    sa = StackedAutoencoder(numpy_rs=numpy_seed,
                            n_visible=n_visible,
                            n_predicted=n_predicted,
                            n_hidden_layers=n_hidden_layers,
                            dropouts=dropouts,
                            hidden_layers=hidden_layers,
                            activation=activation,
                            final_activation=final_activation)

    # train stacked autoencoder
    sa.train(X_train, y_train, X_valid, y_valid, loss=loss,
             optimizer=optimizer, metrics=metrics,
             n_epoch=training_epochs, batch_size=batch_size)

    # get hidden representation using stacked autoencoder
    X_train_enc, X_valid_enc, X_test_enc = \
            sa.get_hidden_representation(X_train, X_valid, X_test)

    # get predictions using the stacked autoencoder for test data
    y_test_sa = sa.predict(X_test)

    # train a random forest classifier using the feature encodings
    print "Training a random forest classifier ..."
    rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
    rfc.fit(X_train_enc, y_train)
    # get predictions using the classifier for test data
    y_test_rfc = rfc.predict(X_test_enc)

    print "Accuracy Random Forests:", accuracy_score(y_test, y_test_rfc)
    print "F1-score Random Forests:", \
                        f1_score(y_test, y_test_rfc, average='weighted')
    print "Accuracy Stacked Autoencoder:", accuracy_score(y_test, y_test_sa)
    print "F1-score Stacked Autoencoder:", \
                        f1_score(y_test, y_test_sa, average='weighted')

if __name__ == '__main__':
    test_stacked_autoencoder()
