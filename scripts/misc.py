import numpy as np
import theano
def binarize_labels(labels, num_classes):
    labels_oh = np.zeros([labels.shape[0], num_classes], dtype=theano.config.floatX)
    for i in xrange(labels.shape[0]):
        labels_oh[i, labels[i]] = 1
    return labels_oh


def filter_words(x, n_words):
    '''
    to filter the words with index > n_words
    predictions:
        type of x is list
        the index of <UNK> in dict is 1
    '''

    new_x = []
    for s in x:
        new_x.append([idx if idx < n_words else 1 for idx in s])

    return new_x
