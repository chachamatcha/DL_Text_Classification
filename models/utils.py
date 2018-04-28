import numpy as np
from tqdm import tqdm # pbar
from keras.preprocessing import text, sequence # text tools
from keras.callbacks import Callback # Callbacks
from sklearn.metrics import roc_auc_score # roc from sklearn


class RocAucEvaluation(Callback):
    '''
    Average column-wise RocAUC callback for Kera CV    
    '''
    def __init__(self, validation_data=(), interval=1):
        '''
        :param validation_data: val data for the AUC calculation
        :param interval: interval of epochs to report the column-wise AUC
        '''
        
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        '''
        :param epoch: current epoch
        :param logs: dictionary log of metrics
        '''
        
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            logs['roc_val'] = score
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            


def tok_embed(X_train, X_test, 
              embedding_path='../data/glove.42B.300d.txt', 
              max_features = 100000, 
              maxlen = 200, 
              embed_size = 300, 
              verbose=True):
    '''
    Tokenizes training and testing data. Creates word emdeddings matrix based on
    test and train.
    
    :param X_train: training data
    :param X_test: test data
    :param max_features: max number of words to store indices of word embeddings
    :param maxlen: max sequence length
    :param embed_size: size(dimensions) of the word embedding file
    :param verbose: verbose flag
    '''
    
    if verbose:
        print('tokenizing...')
    
    # tokenize
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    # pad
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    
    # get word vectors
    embeddings_index = {}
    f = open(embedding_path)
    
    # coefs
    if verbose:
        for line in tqdm(f, desc='getting coefs'):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    else:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs                     
    f.close()

    word_index = tokenizer.word_index

    # init embedding matrix
    num_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embed_size))
    
    if verbose:        
        for word, i in tqdm(word_index.items(),desc='creating embedding matrix'):
            embedding_vector = embeddings_index.get(word)
            if i >= max_features: continue
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    else:
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if i >= max_features: continue
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
                
    # print found word vectors
    if verbose:
        print('Found %s word vectors.' % len(embeddings_index))

    return x_train, x_test, embedding_matrix
