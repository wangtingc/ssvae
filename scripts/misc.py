import numpy as np
import theano
import cPickle as pkl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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


def save_weights(network_params, save_weights_path):
    network_params = [p.get_value(borrow=True) for p in network_params]
    print 'Saving parameters to ' + save_weights_path
    pkl.dump(network_params, open(save_weights_path, 'wb'))


def load_weights(network_params, load_weights_path):
    print 'Loading parameters from ' + load_weights_path
    values = pkl.load(open(load_weights_path, 'rb'))
    for p, v in zip(network_params, values):
        p.set_values(v)

def analyze(data, save_path):
    lens = []
    for sample in data:
        lens.append(len(sample))

    n, bins, patchs = plt.hist(lens, 100, facecolor='green',)
    l = plt.plot(bins)
    plt.title(r'$\mathrm{Histogram\ of\ dataset:}\ \mu: %.1f,\ \sigma, %.1f$'%(np.mean(lens), np.std(lens)))
    plt.savefig(save_path)


