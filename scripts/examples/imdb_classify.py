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
from utils.batchiterator import BatchIterator
from utils.misc import *
import os

from models.rnnclf import RnnClf
from utils.load_data import load_imdb

from datetime import datetime

def init_configurations():
    params = {}
    params['exp_name'] = 'clf'
    params['data'] = 'imdb'
    params['data_path'] = '../../data/proc/imdb/imdb_u.pkl.gz' # to be tested
    params['dict_path'] = '../../data/proc/imdb/imdb_u.dict.pkl.gz'
    #params['emb_path'] = '../../data/proc/imdb/imdb_emb_u.pkl.gz'
    params['emb_path'] = None
    params['batch_size'] = 100
    params['num_classes'] = 2
    params['dim_emb'] = 300
    params['num_units'] = 512
    params['num_samples_train'] = 20000 # the first n samples in trainset.
    params['epoch'] = 200
    params['dev_period'] = 1 # temporary exclude validset
    params['test_period'] = 1
    params['lr'] = 0.0002
    params['num_words'] = 20000
    params['dropout'] = 0.25 # set 0 to no use
    params['exp_time'] = datetime.now().strftime('%m%d%H%M')
    params['save_dir'] = '../../results/' + params['exp_name'] + '_' + params['exp_time']
    params['save_weights_path'] = params['save_dir'] + '/weights.pkl'

    params['pretrain_load_path'] = '../../data/proc/imdb/pretrain_lm1.pkl'
    params['use_final'] = True
    return params


def configuration2str(params):
    s = '[*] printing experiment configuration' + '\n'
    for k in params:
        s += '\t[-] ' + k + ': ' + str(params[k]) + '\n'
    s += '\n'
    return s


def train(params):
    model = RnnClf(n_words = params['num_words'],
                   dim_emb = params['dim_emb'],
                   num_units = params['num_units'],
                   n_classes = params['num_classes'],
                   dropout = params['dropout'],
                   use_final = params['use_final'],
                   lr = params['lr'],
                   pretrain = params['pretrain_load_path'],
                   )

    f_train = model.get_f_train()
    f_test = model.get_f_test()

    train, dev, test, unlabel, wdict, w_emb = load_imdb(params)

    iter_train = BatchIterator(params['num_samples_train'], params['batch_size'], data = train, testing = False)
    iter_dev = BatchIterator(params['num_samples_dev'], params['batch_size'], data = dev, testing = True)
    iter_test = BatchIterator(params['num_samples_test'], params['batch_size'], data = test, testing = True)

    n_batches_train = params['num_samples_train'] / params['batch_size']
    n_batches_dev = params['num_samples_dev'] / params['batch_size']
    n_batches_test = params['num_samples_test'] / params['batch_size']
    #n_batches_train = 1
    #n_batches_dev = 1
    #n_batches_test = 1

    # print the results
    if not os.path.exists(params['save_dir']):
        os.makedirs(params['save_dir'])
        of = open(params['save_dir'] + os.sep + 'log.txt', 'w')
        of.write(configuration2str(params))
        pkl.dump(params, open(params['save_dir'] + '/config.pkl', 'wb'))
        df = open(params['save_dir'] + os.sep + 'details.txt', 'w')
    else:
        raise 'the directory exists: %s' % params['save_dir']


    train_epoch_costs = []
    train_epoch_accs = []
    dev_epoch_accs = []
    test_epoch_accs = []
    print(configuration2str(params))
    for epoch in xrange(params['epoch']):
        epoch_info = '[*] epoch %d' % epoch
        of.write(epoch_info + '\n')
        df.write(epoch_info + '\n')
        print('Epoch:', epoch)

        train_costs = []
        time_costs = []
        train_accs = []
        for batch in xrange(n_batches_train):
            time_s = time.time()
            x, y = iter_train.next()
            x, m = prepare_data(x)
            train_cost, train_acc = f_train(x, m, y)
            train_costs.append(train_cost)
            train_accs.append(train_acc)
            time_costs.append(time.time() - time_s)
            df.write('\t[-] train: batch %d, cost %4f, acc %2f\n' % (batch, train_cost, train_acc))

        train_epoch_costs.append(np.mean(train_costs))
        train_epoch_accs.append(np.mean(train_accs))
        print('train_cost.mean()', np.mean(train_costs))
        print('train_acc.mean()', np.mean(train_accs))
        print('time_cost.mean()', np.mean(time_costs))
        
        dev_accs = []
        if (epoch + 1) % params['dev_period'] == 0:
            for batch in xrange(n_batches_dev):
                x, y = iter_dev.next()
                x, m = prepare_data(x)
                dev_acc = f_test(x, m, y)
                dev_accs.append(dev_acc)
                df.write('\t[-] valid: batch %d, acc %2f\n' % (batch, dev_acc))

        dev_epoch_accs.append(np.mean(dev_accs))
        print('dev_accuracy.mean()', np.mean(dev_accs))

        test_accs = []
        if (epoch + 1) % params['test_period'] == 0:
            for batch in xrange(n_batches_test):
                x, y = iter_test.next()
                x, m = prepare_data(x)
                test_acc = f_test(x, m, y)
                test_accs.append(test_acc)
                df.write('\t[-] test: batch %d, acc %2f\n' % (batch, test_acc))

        test_epoch_accs.append(np.mean(test_accs))
        print('test_accuracy.mean()', np.mean(test_accs))

        # mkdir
        if not os.path.exists(params['save_dir']):
            os.mkdir(params['save_dir'])

        #save the curve
        curve_fig = plt.figure()
        plt.plot(train_epoch_accs, 'r--', label='train')
        plt.plot(dev_epoch_accs, 'bs', label='dev')
        plt.plot(test_epoch_accs, 'g^', label='test')
        plt.legend()
        curve_fig.savefig(os.path.join(params['save_dir'], params['exp_name'] + '.png'))
        plt.close()

        # save configurations
        config_file_path = params['save_dir'] + os.sep + 'config.log'
        with open(config_file_path, 'w') as f:
            for k in params.keys():
                f.write(k + ': ' + str(params[k]) + '\n')
        config_file_path = params['save_dir'] + os.sep + 'config.pkl'
        with open(config_file_path, 'w') as f:
            pkl.dump(params, f)        # save the results

        results = {}
        results['train_costs'] = train_epoch_costs
        results['train_accs'] = train_epoch_accs
        results['dev_accs'] = dev_epoch_accs
        results['test_accs'] = test_epoch_accs
        results_file_path = params['save_dir'] + os.sep + 'results.pkl'
        pkl.dump(results, open(results_file_path, 'wb'))

        # save the weight
        if params['save_weights_path']:
            network_params = model.get_params()
            save_weights(network_params, params['save_weights_path'])



params = init_configurations()
train(params)

