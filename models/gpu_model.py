'''
Collection of Text Classification Keras Algorithms for Toxic Comment Classification Challenge.
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
'''

# Keras
from keras.layers import (Dense, Input, Bidirectional, Activation, Dropout, Embedding, Flatten, CuDNNLSTM, CuDNNGRU,
                          Conv2D, MaxPool2D, concatenate, K, Reshape, LSTM)
from keras.models import Model
from keras import regularizers
from keras.utils import multi_gpu_model



def blstm_2dcnn(maxlen, max_features, embed_size, embedding_matrix,
               embedding_dropout = .5,
               blstm_units = 300,
               blstm_dropout = .2,
               cnn_filters = 100,
               cnn_kernel_size = (5,5),
               max_pool_size = (5,5),
               dense_dropout = .4,
               l2_reg = .00001,
               gpus = 1):
    
    '''
    Bidirectional LSTM with Two-dimensional Max Pooling
    
    :param maxlen: max length of sequence
    :param max_features: max number of word embeddings
    :param embed_size: dimension of word embeddings
    :param embedding_matrix: embedding matrix created from embed file
    :param embedding_dropout: dropout after embedding layer
    :param blstm_units: number of lstm units for the biderectional lstm
    :param blstm_dropout: dropout after the blstm layer
    :param cnn_filters: number of CNN filters
    :param cnn_kernel_size: kernel size of the convolution
    :param max_pool_size: max pool size
    :param dense_dropout: dropout before dense layer
    :param l2_reg: l2 kernel regularizer parameter
    :gpus: number of gpus
    :returns: Keras parallel model
    '''
    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], input_length=maxlen)(inp)
    x = Dropout(embedding_dropout)(x)    

    x = Bidirectional(CuDNNLSTM(blstm_units, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(blstm_dropout)(x)
    x = Reshape((maxlen, blstm_units, 1))(x)

    x = Conv2D(cnn_filters, kernel_size=cnn_kernel_size, padding='valid', kernel_initializer='glorot_uniform')(x)
    x = MaxPool2D(pool_size=max_pool_size)(x)

    x = Flatten()(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(6, activation = "sigmoid",  kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    parallel_model = Model(inputs = inp, outputs = x)
    parallel_model = multi_gpu_model(parallel_model, gpus=gpus)
    parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return parallel_model



def bgru_2dcnn(maxlen, max_features, embed_size, embedding_matrix,
               embedding_dropout = .5,
               bgru_units = 300,
               bgru_dropout = .2,
               cnn_filters = 100,
               cnn_kernel_size = (5,5),
               max_pool_size = (5,5),
               dense_dropout = .4,
               l2_reg = .00001,
               gpus = 1):
    
    '''
    Bidirectional GRU with Two-dimensional Max Pooling
    
    :param maxlen: max length of sequence
    :param max_features: max number of word embeddings
    :param embed_size: dimension of word embeddings
    :param embedding_matrix: embedding matrix created from embed file
    :param embedding_dropout: dropout after embedding layer
    :param bgru_units: number of gru units for the biderectional gru
    :param bgru_dropout: dropout after the bgru layer
    :param cnn_filters: number of cnn filters
    :param cnn_kernel_size: kernel size of the convolution
    :param max_pool_size: max pool size
    :param dense_dropout: dropout before dense layer
    :param l2_reg: l2 kernel regularizer parameter
    :gpus: number of gpus
    :returns: Keras parallel model
    '''
    
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], input_length=maxlen)(inp)
    x = Dropout(embedding_dropout)(x)    

    x = Bidirectional(CuDNNGRU(bgru_units, return_sequences=True), merge_mode='sum')(x)
    x = Dropout(bgru_dropout)(x)
    x = Reshape((maxlen, bgru_units, 1))(x)

    x = Conv2D(cnn_filters, kernel_size=cnn_kernel_size, padding='valid', kernel_initializer='glorot_uniform')(x)
    x = MaxPool2D(pool_size=max_pool_size)(x)

    x = Flatten()(x)
    x = Dropout(dense_dropout)(x)
    x = Dense(6, activation = "sigmoid",  kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    parallel_model = Model(inputs = inp, outputs = x)
    parallel_model = multi_gpu_model(parallel_model, gpus=gpus)
    parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return parallel_model

