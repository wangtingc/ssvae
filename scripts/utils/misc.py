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
    for i in x:
        new_xi = []
        for j in i:
            new_xi.append([idx if idx < n_words else 1 for idx in j])
        new_x.append(new_xi)

    return new_x


def save_weights(network_params, save_weights_path):
    network_params = [p.get_value(borrow=True) for p in network_params]
    print 'Saving parameters to ' + save_weights_path
    pkl.dump(network_params, open(save_weights_path, 'wb'))


def load_weights(network_params, load_weights_path):
    print 'Loading parameters from ' + load_weights_path
    values = pkl.load(open(load_weights_path, 'rb'))
    for p, v in zip(network_params, values):
        p.set_value(v)


def analyze(data, save_path):
    lens = []
    for sample in data:
        lens.append(len(sample))

    n, bins, patchs = plt.hist(lens, 100, facecolor='green',)
    l = plt.plot(bins)
    plt.title(r'$\mathrm{Histogram\ of\ dataset:}\ \mu: %.1f,\ \sigma, %.1f$'%(np.mean(lens), np.std(lens)))
    plt.savefig(save_path)


def prepare_data(x, n_seq = None, l_seq = None):
    '''
    this function calculate x and mask. if n_seq is given, then extract
    n_seqs sentences. if l_seq is given, then extract sentences shorter
    than l_seq
    '''
    assert not(n_seq and l_seq)
    # flatten sentences or select n sentences
    new_x = []
    for i in x:
        if n_seq and len(i) > n_seq:
            len_seq = len(i)
            r = np.random.randint(0, len_seq - n_seq + 1)
            s = []
            for j in i[r: r + n_seq]:
                s += j
            new_x.append(s)
        elif l_seq:
            len_seq = len(i)
            r = np.random.randint(0, len_seq)
            s = i[r]
            while r+1 < len(i) and len(s) + len(i[r+1]) < l_seq:
                s = s + i[r+1]
                r += 1
            new_x.append(s)
        else:
            s = []
            for j in i:
                s += j
            new_x.append(s)

    x = new_x

    x_len = []
    max_len = 0
    for s in x:
        x_len.append(len(s))
        if max_len < len(s):
            max_len = len(s)

    # append <EOS> to data
    xx = np.zeros([len(x), max_len + 1], dtype='int32')
    m = np.zeros([len(x), max_len + 1], dtype=theano.config.floatX)
    for i, s in enumerate(x):
        xx[i, :x_len[i]] = x[i]
        m[i, :x_len[i] + 1] = 1

    if (n_seq or l_seq) and max_len > 200:
        xx = xx[:, :300]
        m = m[:, :300]
    '''
    if not(n_seq or l_seq) and max_len > 1000:
        xx = xx[:, :1000]
        m = m[:, :1000]
    '''

    return xx, m

