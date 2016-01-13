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
from semi_vae import SemiVAE
from misc import *
import os

import matplotlib
matplotlib.use('Agg') # not to use X-desktop
import matplotlib.pyplot as plt

def init_configurations():
    params = {}
    params['data'] = 'imdb'
    params['data_path'] = '../data/proc/imdb_u.pkl.gz' # to be tested
    params['dict_path'] = '../data/proc/imdb_u.dict.pkl.gz'
    params['emb_path'] = '../data/proc/imdb_emb_u.pkl.gz'
    params['num_batches_train'] = 2500
    params['batch_size'] = 100 # for testing and dev
    params['num_classes'] = 2
    params['dim_z'] = 128
    params['num_units_hidden_common'] = 500
    params['num_units_hidden_rnn'] = 128
    params['num_samples_label'] = 22500 # the first n samples in trainset.
    params['epoch'] = 1
    params['valid_period'] = 1 # temporary exclude validset
    params['test_period'] = 1
    params['alpha'] = 0.1
    params['learning_rate'] = 0.0001
    params['num_words'] = 10000
    params['dropout'] = 0.0 # to be tested
    params['exp_name'] = 'semi_lstm_nodropout'
    return params


def load_data(params):
    # train: labelled data for training
    # dev: labelled data for dev
    # test: labelled data for testing
    # unlabel: unlabelled data for semi-supervised learning
    if params['data'] == 'imdb':
        print('loading data from ', params['data_path'])
        with gzip.open(params['data_path']) as f:
            # origin dataset has no devset
            train = pkl.load(f)
            test = pkl.load(f)
            unlabel = pkl.load(f)

            # split devset from train set
            valid_portion = 0.10 # only for imdb dataset (no devset)
            train_pos_idx = np.where(np.asarray(train[1]) ==  1)[0] # np.where return tuple 
            train_neg_idx = np.where(np.asarray(train[1]) ==  0)[0]
            train_pos = [train[0][i] for i in train_pos_idx]
            train_neg = [train[0][i] for i in train_neg_idx]
            num_samples_train_pos = int(len(train_pos) * (1 - valid_portion))
            num_samples_train_neg = int(len(train_neg) * (1 - valid_portion))
            # last n_dev samples from trainset
            dev_pos = [train_pos[i] for i in xrange(num_samples_train_pos, len(train_pos))]
            dev_neg = [train_neg[i] for i in xrange(num_samples_train_neg, len(train_neg))]
            # first #n_label for train, otherwise unlabelled
            assert params['num_samples_label'] % 2 == 0
            num_samples_per_label = params['num_samples_label'] / 2
            assert num_samples_per_label <= num_samples_train_pos
            assert num_samples_per_label <= num_samples_train_neg
            train_pos_2_unlabel = [train_pos[i] for i in xrange(num_samples_per_label, num_samples_train_pos)]
            train_neg_2_unlabel = [train_neg[i] for i in xrange(num_samples_per_label, num_samples_train_neg)]

            train_pos = [train_pos[i] for i in xrange(num_samples_per_label)]
            train_neg = [train_neg[i] for i in xrange(num_samples_per_label)]

            dev_x = dev_pos + dev_neg
            dev_y = [1] * len(dev_pos) + [0] * len(dev_neg)
            dev = [dev_x, binarize_labels(np.asarray(dev_y), params['num_classes']).tolist()]

            train_x = train_pos + train_neg
            train_y = [1] * len(train_pos) + [0] * len(train_neg)
            train = [train_x, binarize_labels(np.asarray(train_y), params['num_classes']).tolist()]

            unlabel = [unlabel + train_pos_2_unlabel + train_neg_2_unlabel]
            test = [test[0], binarize_labels(np.asarray(test[1]), params['num_classses']).tolist()]

            print len(train[0]), len(train[1])
            print len(dev[0]), len(dev[1])
            print len(test[0]), len(test[1])
            print len(unlabel[0])

            params['num_samples_train'] = len(train[0])
            params['num_samples_dev'] = len(dev[0])
            params['num_samples_test'] = len(test[0])
            params['num_samples_unlabel'] = len(unlabel[0])
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
                # use full dictionary if num_words is not set
                if params['num_words'] is None:
                    params['num_words'] = w_emb.shape[0]
                f.close()
        else:   
            # use full dictionary  if num_words is not set
            print('initialize word embedding')
            if params['num_words'] is None:
                max_idx = 0
                for i in wdict.values():
                    max_idx = i if max_idx < i else max_idx
                params['num_words'] = max_idx + 1
            params['dim_emb'] = 100 # set manually
            w_emb = np.random.rand([params['num_words'], params['dim_emb']])
    else:
        # other dataset, e.g. newsgroup and sentiment PTB
        pass
    
    # filter the dict for all dataset
    assert params['num_words'] <= w_emb.shape[0]
    if params['num_words'] != w_emb.shape[0]:
        train = [filter_words(train[0], params['num_words']), train[1]]
        dev = [filter_words(dev[0], params['num_words']), dev[1]]
        test = [filter_words(test[0], params['num_words']), test[1]]
        unlabel = [filter_words(unlabel[0], params['num_words'])]
        w_emb = w_emb[: params['num_words'], :]

    return train, dev, test, unlabel, wdict, w_emb


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


def build_model(params, w_emb):
    l_x = layers.InputLayer((None, None))
    l_m = layers.InputLayer((None, None))
    l_y = layers.InputLayer((None, params['num_classes']))
    
    #debug
    #params['num_samples_train'] = 22500
    beta = params['alpha'] * params['num_samples_train'] / params['num_samples_label']
    semi_vae = SemiVAE([l_x, l_m, l_y],
                       w_emb,
                       params['num_units_hidden_common'],
                       params['dim_z'],
                       beta,
                       params['num_units_hidden_rnn'],
                       )

    x_l = T.imatrix()
    m_l = T.matrix()
    y_l = T.matrix()
    x_u = T.imatrix()
    m_u = T.matrix()
    cost = semi_vae.get_cost_together([x_l, m_l, y_l, x_u, m_u])
    acc = semi_vae.get_cost_test([x_l, m_l, y_l])

    network_params = semi_vae.get_params()

    for param in network_params:
        print param.get_value().shape, param.name

    params_update = updates.rmsprop(cost, network_params, params['learning_rate'])

    #f_train =theano.function([x, m], pred)
    f_train = theano.function([x_l, m_l, y_l, x_u, m_u], cost, updates = params_update)
    f_test = theano.function([x_l, m_l, y_l], acc)

    return f_train, f_test


def train(params):
    train, dev, test, unlabel, wdict, w_emb = load_data(params)
    #import numpy as np
    #w_emb = np.random.rand(200000, 100).astype('float32')
    f_train, f_test = build_model(params, w_emb)
    
    assert params['num_samples_train'] % params['num_batches_train'] == 0
    assert params['num_samples_unlabel'] % params['num_batches_train'] == 0
    assert params['num_samples_dev'] % params['batch_size'] == 0
    assert params['num_samples_test'] % params['batch_size'] == 0
    
    # semi = {train, unlabel}
    num_batches_train = params['num_batches_train']
    num_batches_dev = params['num_samples_dev'] / params['batch_size']
    num_batches_test = params['num_samples_test'] / params['batch_size']

    print num_batches_train, num_batches_dev, num_batches_test

    batch_size_l = params['num_samples_train'] / num_batches_train
    batch_size_u = params['num_samples_unlabel'] / num_batches_train

    print num_batches_train, num_batches_dev, num_batches_test

    iter_train = BatchIterator(params['num_samples_train'], batch_size_l, data = train)
    iter_unlabel = BatchIterator(params['num_samples_unlabel'], batch_size_u, data = unlabel)
    iter_dev = BatchIterator(params['num_samples_dev'], params['batch_size'], data = dev)
    iter_test = BatchIterator(params['num_samples_test'], params['batch_size'], data = test)
    
    train_epoch_costs = []
    train_epoch_accs = []
    dev_epoch_accs = []
    test_epoch_accs = []

    for epoch in xrange(params['epoch']):
        print('Epoch:', epoch)
        #n_batches_train = 1
        #n_batches_test = 1

        train_costs = []
        time_costs = []
        train_accs = []
        #debug
        num_batches_train = 1
        num_batches_dev = 1
        num_batches_test = 1
        for batch in xrange(num_batches_train):
            time_s = time.time()
            x_l, y_l = iter_train.next()
            x_l, m_l = prepare_data(x_l)
            x_u = iter_unlabel.next()[0]
            x_u, m_u = prepare_data(x_u)
            train_cost = f_train(x_l, m_l, y_l, x_u, m_u)
            print train_cost
            train_acc = f_test(x_l, m_l, y_l)
            train_costs.append(train_cost)
            train_accs.append(train_acc)
            time_costs.append(time.time() - time_s)

        train_epoch_costs.append(np.mean(train_costs))
        train_epoch_accs.append(np.mean(train_accs))
        print('train_cost.mean()', np.mean(train_costs))
        print('train_acc.mean()', np.mean(train_accs))
        print('time_cost.mean()', np.mean(time_costs))
        
        dev_accs = []
        for batch in xrange(num_batches_dev):
            x, y = iter_dev.next()
            x, m = prepare_data(x)
            dev_acc = f_test(x, m, y)
            dev_accs.append(dev_acc)
    
        dev_epoch_accs.append(np.mean(dev_accs))
        print('dev_accuracy.mean()', np.mean(dev_accs))

        test_accs = []
        for batch in xrange(num_batches_test):
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

