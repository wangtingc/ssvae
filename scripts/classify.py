import time
import theano
import theano.tensor as T
import numpy as np
import lasagne.layers as layers
import lasagne.nonlinearities as nonlinearities
import lasagne.objectives as objectives
import lasagne.updates  as updates
import gzip
import cPickle as pkl
from batchiterator import BatchIterator

def init_configurations():
    params = {}
    params['data'] = 'imdb'
    params['data_path'] = '../data/proc/imdb_l.pkl.gz'
    params['dict_path'] = '../data/proc/imdb_l.dict.pkl.gz'
    params['emb_path'] = '../data/proc/imdb_emb.pkl.gz'
    params['batch_size'] = 32
    params['n_classes'] = 2
    params['dim_z'] = 128
    params['nm_units_hidden_common'] = 500
    params['num_samples_train_label'] = 1000 # the first n samples in trainset.
    params['epoch'] = 100
    params['valid_period'] = 1 # temporary exclude validset
    params['test_period'] = 1
    params['alpha'] = 0.1
    params['learning_rate'] = 0.0001
    return params


def load_data(params):
    if params['data'] == 'imdb':
        with gzip.open(params['data_path']) as f:
            train = pkl.load(f)
            test = pkl.load(f)
            params['n_samples_train'] = len(train[0])
            params['n_samples_test'] = len(test[0])

        with gzip.open(params['dict_path']) as f:
            wdict = pkl.load(f)

        if params['emb_path']:
            with gzip.open(params['emb_path']) as f:
                w_emb = pkl.load(f).astype(theano.config.floatX)
                params['n_words'] = w_emb.shape[0]
                params['dim_emb'] = w_emb.shape[1]
        else:
            max_idx = 0
            for i in wdict.values():
                max_idx = i if max_idx < i else max_idx
            params['n_words'] = max_idx + 1
            params['dim_emb'] = 100
            w_emb = np.random.rand([params['n_words'], params['dim_emb']])

    return train, test, wdict, w_emb


def prepare_data(x):
    x_len = []
    max_len = 0
    for s in x:
        x_len.append(len(s))
        if max_len < len(s):
            max_len = len(s)

    xx = np.zeros([len(x), max_len + 1], dtype='int32')
    m = np.zeros([len(x), max_len + 1], dtype=theano.config.floatX)
    for i, s in enumerate(x):
        xx[i, :x_len[i]] = x[i]
        m[i, :x_len[i] + 1] = 1

    return xx, m



class MeanLayer(layers.MergeLayer):
    def __init__(self, incomings):
        super(MeanLayer, self).__init__(incomings)

    def get_output_for(self, inputs):
        h, m = inputs
        o = (h * m[:, :, None]).sum(axis=1)
        o = o / m.sum(axis=1)[:, None]
        return o

    def get_params(self):
        return []

    def get_output_shape_for(self, input_shapes):
        return [input_shapes[0][0], input_shapes[0][2]]


def build_model(params, w_emb):
    x = T.imatrix()
    m = T.matrix()
    y = T.ivector()
    l_in = layers.InputLayer((params['batch_size'], None))
    l_mask = layers.InputLayer((params['batch_size'],  None))
    l1 = layers.EmbeddingLayer(l_in, params['n_words'], params['dim_emb'], W=w_emb)
    embs = l1.get_output_for(x)
    f_embs = theano.function([x], embs)
    #print f_embs([[1,2], [2,2]])
    l2 = layers.LSTMLayer(l1, params['dim_z'], mask_input = l_mask)
    h = l2.get_output_for([embs, m])
    l3 = MeanLayer((l2, l_mask))
    h_mean = l3.get_output_for((h, m))
    l4 = layers.DenseLayer(l3, params['n_classes'], nonlinearity = nonlinearities.softmax)
    pred = l4.get_output_for(h_mean)

    cost = objectives.categorical_crossentropy(pred, y).mean()
    acc = T.eq(T.argmax(pred, axis=1), y).mean()
    network_params = layers.get_all_params(l4)

    for param in network_params:
        print param.get_value().shape, param.name

    params_update = updates.adam(cost, network_params, params['learning_rate'])

    #f_train =theano.function([x, m], pred)
    f_train = theano.function([x, m, y], cost, updates = params_update)
    f_test = theano.function([x, m, y], acc)

    return f_train, f_test


def train(params):
    train, test, wdict, w_emb = load_data(params)
    f_train, f_test = build_model(params, w_emb)

    iter_train = BatchIterator(range(params['n_samples_train']), params['batch_size'], data = train)
    iter_test = BatchIterator(range(params['n_samples_test']), params['batch_size'], data = test)

    for epoch in xrange(params['epoch']):
        n_batches_train = params['n_samples_train'] / params['batch_size']
        n_batches_test = params['n_samples_test'] / params['batch_size']
        #n_batches_train = 1
        #n_batches_test = 1

        train_costs = []
        time_costs = []
        for batch in xrange(n_batches_train):
            time_s = time.time()
            x, y = iter_train.next()
            x, m = prepare_data(x)
            train_cost = f_train(x, m, y)
            train_costs.append(train_cost)
            time_costs.append(time.time() - time_s)
        print('train_cost.mean()', np.mean(train_costs))
        print('time_cost.mean()', np.mean(time_costs))

        test_accs = []
        for batch in xrange(n_batches_test):
            x, y = iter_test.next()
            x, m = prepare_data(x)
            test_acc = f_test(x,m,  y)
            test_accs.append(test_acc)
        print('test_accuracy.mean()', np.mean(test_accs))






params = init_configurations()
train(params)

