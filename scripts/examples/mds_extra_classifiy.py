import sys
sys.path.append('../')

import matplotlib
matplotlib.use('Agg') # not to use X-desktop
import matplotlib.pyplot as plt
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
from datetime import datetime
import os

from utils.batchiterator import BatchIterator
from utils.misc import *
from utils.load_data import *
from models.semi_mds import SemiMDS

def init_configurations():
    params = {}
    params['exp_name'] = 'clf_mdsex_6k'
    params['data'] = 'mds'
    params['data_path'] = '../../data/proc/mds/mds.pkl.gz' # to be tested
    params['dict_path'] = '../../data/proc/mds/mds.dict.pkl.gz'
    #params['emb_path'] = '../data/proc/imdb_emb_u.pkl.gz'
    params['emb_path'] = None
    params['num_batches_train'] = 100
    params['batch_size'] = 100 # for testing and dev
    params['num_classes'] = 2
    params['num_domains'] = 4
    params['dim_z'] = 50
    params['num_units_hidden_common'] = 100
    params['num_units_hidden_rnn'] = 512
    params['num_samples_train'] = 5600 # the first n samples in trainset.
    params['epoch'] = 200
    params['valid_period'] = 10 # temporary exclude validset
    params['test_period'] = 10
    params['alpha'] = 0.2
    params['learning_rate'] = 0.0001
    params['num_words'] = 20000
    params['weight_decay_rate'] = 2e-6
    params['annealing_center'] = 90
    params['annealing_width'] = 10
    params['exp_time'] = datetime.now().strftime('%m%d%H%M')
    params['save_directory'] = '../../results/semi_mds_' + params['exp_time']
    params['save_weights_path'] = params['save_directory'] + '/weights.pkl'
    params['load_weights_path'] = None
    params['num_seqs'] = None
    params['len_seqs'] = 200
    params['word_dropout'] = 0.0
    params['dropout'] = 0.2
    return params


def build_model(params, w_emb):
    #debug
    #params['num_samples_train'] = 22500
    beta = params['alpha'] * (params['num_samples_train'] + params['num_samples_unlabel']) / params['num_samples_train']
    print('beta', beta)
    semi_mds = SemiMDS(w_emb,
                       params['num_classes'],
                       params['num_domains'],
                       params['num_units_hidden_common'],
                       params['dim_z'],
                       beta,
                       params['num_units_hidden_rnn'],
                       params['weight_decay_rate'],
                       params['dropout'],
                       params['word_dropout'],
                       )

    x_l_all = T.imatrix()
    m_l_all = T.matrix()
    x_l_sub = T.imatrix()
    m_l_sub = T.matrix()
    d_l = T.matrix()
    y_l = T.matrix()
    x_u_all = T.imatrix()
    m_u_all = T.matrix()
    x_u_sub = T.imatrix()
    m_u_sub = T.matrix()
    kl_w = T.scalar()

    inputs_l = [x_l_all, m_l_all, x_l_sub, m_l_sub, y_l]
    inputs_u = [x_u_all, m_u_all, x_u_sub, m_u_sub]

    # debug
    emb_l_all = semi_mds.embed_layer.get_output_for(x_l_all)
    cost, acc_train = semi_mds.get_cost_C([x_l_all, emb_l_all, m_l_all, d_l, y_l])
    acc = semi_mds.get_cost_test([x_l_all, m_l_all, d_l, y_l])

    network_params = layers.get_all_params(semi_mds.classifier)

    if params['load_weights_path']:
        load_weights(semi_mds.get_params(), params['load_weights_path'])


    for param in network_params:
        print param.get_value().shape, param.name

    params_update = updates.adam(cost.mean(), network_params, params['learning_rate'])

    #f_train =theano.function([x, m], pred)
    print inputs_l + inputs_u
    f_train = theano.function([x_l_all, m_l_all, d_l, y_l], [cost, acc_train], updates = params_update)
    f_test = theano.function([x_l_all, m_l_all, d_l, y_l], acc)

    return semi_mds, f_train, f_test


def train(params):
    train, dev, test, unlabel, wdict, w_emb = load_mds(params)

    #import numpy as np
    #w_emb = np.random.rand(200000, 100).astype('float32')
    semi_mds, f_train, f_test = build_model(params, w_emb)

    #assert params['num_samples_train'] % params['num_batches_train'] == 0
    #assert params['num_samples_unlabel'] % params['num_batches_train'] == 0
    assert params['num_samples_dev'] % params['batch_size'] == 0
    assert params['num_samples_test'] % params['batch_size'] == 0

    num_batches_train = params['num_batches_train']
    num_batches_dev = params['num_samples_dev'] / params['batch_size']
    num_batches_test = params['num_samples_test'] / params['batch_size']

    print(num_batches_train, num_batches_dev, num_batches_test)

    batch_size_l = params['num_samples_train'] / num_batches_train
    batch_size_u = params['num_samples_unlabel'] / num_batches_train

    print(num_batches_train, num_batches_dev, num_batches_test)

    testing = False if params['num_seqs'] or params['len_seqs'] else True
    iter_train = BatchIterator(params['num_samples_train'], batch_size_l, data = train, testing = testing)
    iter_unlabel = BatchIterator(params['num_samples_unlabel'], batch_size_u, data = unlabel, testing = testing)
    iter_dev = BatchIterator(params['num_samples_dev'], params['batch_size'], data = dev, testing = testing)
    iter_test = BatchIterator(params['num_samples_test'], params['batch_size'], data = test, testing = testing)

    train_epoch_costs = []
    train_epoch_accs = []
    dev_epoch_accs = []
    test_epoch_accs = []

    for epoch in xrange(params['epoch']):
        print('Epoch:', epoch)

        train_costs = []
        time_costs = []
        train_accs = []
        #debug
        #num_batches_train = 1
        #num_batches_dev = 1
        #num_batches_test = 1
        for batch in xrange(num_batches_train):
            time_s = time.time()
            x_l, y_l, d_l = iter_train.next()

            x_l_all, m_l_all = prepare_data(x_l)
            inputs_l = [x_l_all, m_l_all, d_l, y_l]

            #print x_l_all.shape, x_u_all.shape
            #print x_l_sub.shape, x_u_sub.shape
            train_cost, train_acc = f_train(*inputs_l)
            y_l = np.asarray(y_l, dtype=theano.config.floatX)
            #print time.time() - time_s
            train_costs.append(train_cost)
            train_accs.append(train_acc)
            time_costs.append(time.time() - time_s)

        train_epoch_costs.append(np.mean(train_costs))
        train_epoch_accs.append(np.mean(train_accs))
        print('train_cost.mean()', np.mean(train_costs))
        print('time_cost.mean()', np.mean(time_costs))
        print('train_acc.mean()', np.mean(train_accs))

        dev_accs = []
        for batch in xrange(num_batches_dev):
            x, y, d = iter_dev.next()
            x_all, m_all = prepare_data(x)
            dev_acc = f_test(x_all, m_all, d, y)
            dev_accs.append(dev_acc)

        dev_epoch_accs.append(np.mean(dev_accs))
        print('dev_acc.mean()', np.mean(dev_accs))

        test_accs = []
        for batch in xrange(num_batches_test):
            x, y, d = iter_test.next()
            x, m = prepare_data(x)
            test_acc = f_test(x, m, d, y)
            test_accs.append(test_acc)

        test_epoch_accs.append(np.mean(test_accs))
        print('test_accuracy.mean()', np.mean(test_accs))

        # mkdir
        if not os.path.exists(params['save_directory']):
            os.mkdir(params['save_directory'])

        # save configurations
        config_file_path = params['save_directory'] + os.sep + 'config.log'
        with open(config_file_path, 'w') as f:
            for k in params.keys():
                f.write(k + ': ' + str(params[k]) + '\n')

        # save the curve
        curve_fig = plt.figure()
        plt.plot(train_epoch_accs, 'r--', label='train')
        plt.plot(dev_epoch_accs, 'bs', label='dev')
        plt.plot(test_epoch_accs, 'g^', label='test')
        plt.legend()
        curve_fig.savefig(os.path.join(params['save_directory'], params['exp_name'] + '.png'))

        # save the results
        results = {}
        results['train_costs'] = train_epoch_costs
        results['train_accs'] = train_epoch_accs
        results['dev_accs'] = dev_epoch_accs
        results['test_accs'] = test_epoch_accs
        results_file_path = params['save_directory'] + os.sep + 'results.pkl'
        pkl.dump(results, open(results_file_path, 'wb'))

        # save the weight
        if params['save_weights_path']:
            network_params = semi_mds.get_params()
            save_weights(network_params, params['save_weights_path'])

        print('')

params = init_configurations()
train(params)
