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
from misc import *
import os

import matplotlib
matplotlib.use('Agg') # not to use X-desktop
import matplotlib.pyplot as plt

def init_configurations():
    params = {}
    params['data'] = 'imdb'
    params['data_path'] = '../data/proc/imdb_l.pkl.gz' # to be tested
    params['dict_path'] = '../data/proc/imdb_l.dict.pkl.gz'
    params['emb_path'] = '../data/proc/imdb_emb_l.pkl.gz'
    params['batch_size'] = 32
    params['n_classes'] = 2
    params['dim_z'] = 128
    params['num_units_hidden_common'] = 500
    params['num_samples_label'] = 22500 # the first n samples in trainset.
    params['epoch'] = 100
    params['valid_period'] = 1 # temporary exclude validset
    params['test_period'] = 1
    params['alpha'] = 0.1
    params['learning_rate'] = 0.0001
    params['n_words'] = None
    params['dropout'] = 0.5 # set 0 to no use
    params['exp_name'] = 'clf_lstm_dropout'
    return params


def load_data(params):
    if params['data'] == 'imdb':
        print('loading data from ', params['data_path'])
        with gzip.open(params['data_path']) as f:
            # origin dataset has no devset
            train = pkl.load(f)
            test = pkl.load(f)

            # split devset from train set
            valid_portion = 0.1 # only for imdb dataset (no devset)
            train_pos_idx = np.where(np.asarray(train[1]) ==  1)[0] # np.where return tuple 
            train_neg_idx = np.where(np.asarray(train[1]) ==  0)[0]
            train_pos = [train[0][i] for i in train_pos_idx]
            train_neg = [train[0][i] for i in train_neg_idx]
            num_samples_train_pos = int(len(train_pos) * (1 - valid_portion))
            num_samples_train_neg = int(len(train_neg) * (1 - valid_portion))
            # last n_dev samples from trainset
            dev_pos = [train_pos[i] for i in xrange(num_samples_train_pos, len(train_pos))]
            dev_neg = [train_neg[i] for i in xrange(num_samples_train_neg, len(train_neg))]

            train_pos = [train_pos[i] for i in xrange(num_samples_train_pos)]
            train_neg = [train_neg[i] for i in xrange(num_samples_train_neg)]

            dev_x = dev_pos + dev_neg
            dev_y = [1] * len(dev_pos) + [0] * len(dev_neg)
            dev = [dev_x, dev_y]

            train_x = train_pos + train_neg
            train_y = [1] * len(train_pos) + [0] * len(train_neg)
            train = [train_x, train_y]

            print len(train[0]), len(train[1])
            print len(dev[0]), len(dev[1])
            print len(test[0]), len(test[1])

            params['n_samples_train'] = len(train[0])
            params['n_samples_dev'] = len(dev[0])
            params['n_samples_test'] = len(test[0])
            f.close()
    
        print('loading dict from ', params['dict_path'])
        with gzip.open(params['dict_path']) as f:
            wdict = pkl.load(f)
            f.close()

        if params['emb_path']:
            print('loading word embedding from', params['emb_path'])
            with gzip.open(params['emb_path']) as f:
                w_emb = pkl.load(f).astype(theano.config.floatX)
                params['dim_emb'] = w_emb.shape[1]
                # use full dictionary if n_words is not set
                if params['n_words'] is None:
                    params['n_words'] = w_emb.shape[0]
                f.close()
        else:   
            # use full dictionary  if n_words is not set
            print('initialize word embedding')
            if params['n_words'] is None:
                max_idx = 0
                for i in wdict.values():
                    max_idx = i if max_idx < i else max_idx
                params['n_words'] = max_idx + 1
            params['dim_emb'] = 100 # set manually
            w_emb = np.random.rand([params['n_words'], params['dim_emb']])
    else:
        # other dataset, e.g. newsgroup and sentiment PTB
        pass

    
    maxint = 0
    for i in train[0]:
        for j in i:
            if maxint < j:
                maxint = j
    print maxint
    print w_emb.shape[0]

    
    # filter the dict for all dataset
    assert params['n_words'] <= w_emb.shape[0]
    if params['n_words'] != w_emb.shape[0]:
        train = [filter_words(train[0], params['n_words']), train[1]]
        dev = [filter_words(dev[0], params['n_words']), dev[1]]
        test = [filter_words(test[0], params['n_words']), test[1]]
        w_emb = w_emb[: params['n_words'], :]

    return train, dev, test, wdict, w_emb


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
        return (input_shapes[0][0], input_shapes[0][2])


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
    
    # dropout
    l1_dropout = layers.DropoutLayer(l1, params['dropout'])
    embs = l1_dropout.get_output_for(embs)

    l2 = layers.LSTMLayer(l1_dropout, params['dim_z'], mask_input = l_mask)
    h = l2.get_output_for([embs, m])
    l3 = MeanLayer((l2, l_mask))
    h_mean = l3.get_output_for((h, m))
    
    # dropout
    l3_dropout = layers.DropoutLayer(l3, params['dropout'])
    h_mean = l3_dropout.get_output_for(h_mean)

    l4 = layers.DenseLayer(l3_dropout, params['n_classes'], nonlinearity = nonlinearities.softmax)
    pred = l4.get_output_for(h_mean)

    cost = objectives.categorical_crossentropy(pred, y).mean()
    acc = T.eq(T.argmax(pred, axis=1), y).mean()
    network_params = layers.get_all_params(l4)

    for param in network_params:
        print param.get_value().shape, param.name

    params_update = updates.rmsprop(cost, network_params, params['learning_rate'])

    #f_train =theano.function([x, m], pred)
    f_train = theano.function([x, m, y], cost, updates = params_update)
    f_test = theano.function([x, m, y], acc)

    return f_train, f_test


def train(params):
    train, dev, test, wdict, w_emb = load_data(params)
    f_train, f_test = build_model(params, w_emb)

    iter_train = BatchIterator(range(params['n_samples_train']), params['batch_size'], data = train)
    iter_dev = BatchIterator(range(params['n_samples_dev']), params['batch_size'], data = dev)
    iter_test = BatchIterator(range(params['n_samples_test']), params['batch_size'], data = test)
    
    train_epoch_costs = []
    train_epoch_accs = []
    dev_epoch_accs = []
    test_epoch_accs = []
    for epoch in xrange(params['epoch']):
        print('Epoch:', epoch)
        n_batches_train = params['n_samples_train'] / params['batch_size']
        n_batches_dev = params['n_samples_dev'] / params['batch_size']
        n_batches_test = params['n_samples_test'] / params['batch_size']
        #n_batches_train = 1
        #n_batches_test = 1

        train_costs = []
        time_costs = []
        train_accs = []
        for batch in xrange(n_batches_train):
            time_s = time.time()
            x, y = iter_train.next()
            x, m = prepare_data(x)
            train_cost = f_train(x, m, y)
            train_acc = f_test(x, m, y)
            train_costs.append(train_cost)
            train_accs.append(train_acc)
            time_costs.append(time.time() - time_s)

        train_epoch_costs.append(np.mean(train_costs))
        train_epoch_accs.append(np.mean(train_accs))
        print('train_cost.mean()', np.mean(train_costs))
        print('train_acc.mean()', np.mean(train_accs))
        print('time_cost.mean()', np.mean(time_costs))
        
        dev_accs = []
        for batch in xrange(n_batches_dev):
            x, y = iter_dev.next()
            x, m = prepare_data(x)
            dev_acc = f_test(x, m, y)
            dev_accs.append(dev_acc)
    
        dev_epoch_accs.append(np.mean(dev_accs))
        print('dev_accuracy.mean()', np.mean(dev_accs))

        test_accs = []
        for batch in xrange(n_batches_test):
            x, y = iter_test.next()
            x, m = prepare_data(x)
            test_acc = f_test(x, m, y)
            test_accs.append(test_acc)

        test_epoch_accs.append(np.mean(test_accs))
        print('test_accuracy.mean()', np.mean(test_accs))

    #save the curve
    curve_fig = plt.figure()
    plt.plot(train_epoch_accs, 'r--', label='train')
    plt.plot(dev_epoch_accs, 'bs', label='dev')
    plt.plot(test_epoch_accs, 'g^', label='test')
    plt.legend()
    curve_fig.savefig(os.path.join('results', params['exp_name'] + '.png'))


params = init_configurations()
train(params)

