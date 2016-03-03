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
from batchiterator import BatchIterator
from semi_vae import SemiVAE
from misc import *
from datetime import datetime
import os


def init_configurations():
    params = {}
    params['exp_name'] = 'semi_20k_sc_0.2_5k'
    params['data'] = 'imdb'
    params['data_path'] = '../data/proc/imdb_u.pkl.gz' # to be tested
    params['dict_path'] = '../data/proc/imdb_u.dict.pkl.gz'
    #params['emb_path'] = '../data/proc/imdb_emb_u.pkl.gz'
    params['emb_path'] = None
    params['num_batches_train'] = 1250
    params['batch_size'] = 100 # for testing and dev
    params['num_classes'] = 2
    params['dim_z'] = 50
    params['num_units_hidden_common'] = 100
    params['num_units_hidden_rnn'] = 512
    params['num_samples_train'] = 5000 # the first n samples in trainset.
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
    params['save_directory'] = '../results/semi_vae_' + params['exp_time']
    params['save_weights_path'] = params['save_directory'] + '/weights.pkl'
    params['load_weights_path'] = None
    params['num_seqs'] = None
    params['len_seqs'] = 200
    params['word_dropout'] = 0.0
    params['dropout'] = 0.2
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

            np.random.seed(1234567890)
            train_shuffle_idx = np.random.permutation(len(train[0]))
            train_shuffle_x = [train[0][i] for i in train_shuffle_idx]
            train_shuffle_y = [train[1][i] for i in train_shuffle_idx]
            train = [train_shuffle_x, train_shuffle_y]

            # split devset from train set
            valid_portion = 0.20 # only for imdb dataset (no devset)
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
            assert params['num_samples_train'] % 2 == 0
            num_samples_per_label = params['num_samples_train'] / 2
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
            test = [test[0], binarize_labels(np.asarray(test[1]), params['num_classes']).tolist()]

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
            params['dim_emb'] = 200 # set manually
            w_emb = np.random.rand(params['num_words'], params['dim_emb'])
            w_emb = w_emb.astype(theano.config.floatX)
    else:
        # other dataset, e.g. newsgroup and sentiment PTB
        pass

    # filter the dict for all dataset
    assert params['num_words'] <= w_emb.shape[0]
    if not params['emb_path'] or params['num_words'] != w_emb.shape[0]:
        train = [filter_words(train[0], params['num_words']), train[1]]
        dev = [filter_words(dev[0], params['num_words']), dev[1]]
        test = [filter_words(test[0], params['num_words']), test[1]]
        unlabel = [filter_words(unlabel[0], params['num_words'])]
        w_emb = w_emb[: params['num_words'], :]

    return train, dev, test, unlabel, wdict, w_emb


def build_model(params, w_emb):
    l_x = layers.InputLayer((None, None))
    l_m = layers.InputLayer((None, None))
    l_y = layers.InputLayer((None, params['num_classes']))

    #debug
    #params['num_samples_train'] = 22500
    beta = params['alpha'] * (params['num_samples_train'] + params['num_samples_unlabel']) / params['num_samples_train']
    print('beta', beta)
    semi_vae = SemiVAE([l_x, l_m, l_y],
                       w_emb,
                       params['num_units_hidden_common'],
                       params['dim_z'],
                       beta,
                       params['num_units_hidden_rnn'],
                       params['weight_decay_rate'],
                       params['dropout'],
                       )

    x_l_all = T.imatrix()
    m_l_all = T.matrix()
    x_l_sub = T.imatrix()
    m_l_sub = T.matrix()
    y_l = T.matrix()
    x_u_all = T.imatrix()
    m_u_all = T.matrix()
    x_u_sub = T.imatrix()
    m_u_sub = T.matrix()
    kl_w = T.scalar()

    inputs_l = [x_l_all, m_l_all, x_l_sub, m_l_sub, y_l]
    inputs_u = [x_u_all, m_u_all, x_u_sub, m_u_sub]

    # debug
    embs_l_sub = semi_vae.embed_layer.get_output_for(x_l_sub)
    cost_l = semi_vae.get_cost_L([x_l_sub, embs_l_sub, m_l_sub, y_l], kl_w, 0)

    cost, train_acc = semi_vae.get_cost_together(inputs_l, inputs_u, kl_w, params['word_dropout'])
    test_acc = semi_vae.get_cost_test([x_l_all, m_l_all, y_l])

    network_params = semi_vae.get_params()
    if params['load_weights_path']:
        load_weights(network_params, params['load_weights_path'])


    for param in network_params:
        print param.get_value().shape, param.name

    params_update = updates.adam(cost, network_params, params['learning_rate'])

    #f_train =theano.function([x, m], pred)
    print inputs_l + inputs_u
    f_debug = theano.function([x_l_sub, m_l_sub, y_l, kl_w], cost_l)
    f_train = theano.function(inputs_l + inputs_u + [kl_w], [cost, train_acc], updates = params_update)
    f_test = theano.function([x_l_all, m_l_all, y_l], test_acc)

    return semi_vae, f_debug, f_train, f_test


def train(params):
    train, dev, test, unlabel, wdict, w_emb = load_data(params)
    #import numpy as np
    #w_emb = np.random.rand(200000, 100).astype('float32')
    semi_vae, f_debug, f_train, f_test = build_model(params, w_emb)

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
    train_epoch_diffs = []
    train_epoch_accs = []
    train_epoch_ppls = []
    dev_epoch_accs = []
    dev_epoch_ppls = []
    test_epoch_accs = []
    test_epoch_ppls = []

    for epoch in xrange(params['epoch']):
        print('Epoch:', epoch)

        train_costs = []
        train_diffs = []
        time_costs = []
        train_ppls = []
        train_accs = []
        #debug
        #num_batches_train = 1
        #num_batches_dev = 1
        #num_batches_test = 1
        for batch in xrange(num_batches_train):
            time_s = time.time()
            x_l, y_l = iter_train.next()
            x_l_all, m_l_all = prepare_data(x_l)
            x_l_sub, m_l_sub = prepare_data(x_l, params['num_seqs'], params['len_seqs'])
            inputs_l = [x_l_all, m_l_all, x_l_sub, m_l_sub, y_l]

            x_u = iter_unlabel.next()[0]
            x_u_all, m_u_all = prepare_data(x_u)
            x_u_sub, m_u_sub = prepare_data(x_u, params['num_seqs'], params['len_seqs'])
            inputs_u = [x_u_all, m_u_all, x_u_sub, m_u_sub]

            # calculate kl_w
            anneal_value = epoch + np.float32(batch)/num_batches_train - params['annealing_center']
            anneal_value = (anneal_value / params['annealing_width']).astype(theano.config.floatX)
            kl_w = np.float32(1) if anneal_value > 7.0 else 1/(1 + np.exp(-anneal_value))
            kl_w = kl_w.astype(theano.config.floatX)
            # debug
            kl_w = np.float32(0)

            #print x_l_all.shape, x_u_all.shape
            #print x_l_sub.shape, x_u_sub.shape
            train_cost, train_acc = f_train(*(inputs_l + inputs_u + [kl_w]))
            y_l = np.asarray(y_l, dtype=theano.config.floatX)
            cost_l_rig = f_debug(x_l_sub, m_l_sub, y_l, 0)
            cost_l_wro = f_debug(x_l_sub, m_l_sub, 1-y_l, 0)
            #print time.time() - time_s
            train_diff = (cost_l_rig > cost_l_wro).mean()
            train_ppl = np.exp(cost_l_rig.sum() / m_l_sub.sum())
            train_costs.append(train_cost)
            train_diffs.append(train_diff)
            train_ppls.append(train_ppl)
            train_accs.append(train_acc)
            time_costs.append(time.time() - time_s)

        train_epoch_costs.append(np.mean(train_costs))
        train_epoch_diffs.append(np.mean(train_diffs))
        train_epoch_ppls.append(np.mean(train_ppls))
        train_epoch_accs.append(np.mean(train_accs))
        print('train_cost.mean()', np.mean(train_costs))
        print('train_diff.mean()', np.mean(train_diffs))
        print('train_ppl.mean()', np.mean(train_ppls))
        print('time_cost.mean()', np.mean(time_costs))
        print('train_acc.mean()', np.mean(train_accs))

        dev_accs = []
        dev_ppls = []
        for batch in xrange(num_batches_dev):
            x, y = iter_dev.next()
            x_all, m_all = prepare_data(x)
            x_sub, m_sub = prepare_data(x, params['num_seqs'], params['len_seqs'])
            dev_acc = f_test(x_all, m_all, y)
            dev_l = f_debug(x_sub, m_sub, y, 0)
            dev_ppl = np.exp(dev_l.sum() / m_sub.sum())
            dev_accs.append(dev_acc)
            dev_ppls.append(dev_ppl)

        dev_epoch_accs.append(np.mean(dev_accs))
        dev_epoch_ppls.append(np.mean(dev_ppls))
        print('dev_acc.mean()', np.mean(dev_accs))
        print('dev_ppl.mean()', np.mean(dev_ppls))

        test_accs = []
        for batch in xrange(num_batches_test):
            x, y = iter_test.next()
            x, m = prepare_data(x)
            test_acc = f_test(x, m, y)
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
        curve_fig.savefig(os.path.join('../results', params['exp_name'] + '.png'))

        # save the results
        results = {}
        results['train_costs'] = train_epoch_costs
        results['train_diffs'] = train_epoch_diffs
        results['train_accs'] = train_epoch_accs
        results['train_ppls'] = train_epoch_ppls
        results['dev_accs'] = dev_epoch_accs
        results['dev_ppls'] = dev_epoch_ppls
        results['test_accs'] = test_epoch_accs
        results_file_path = params['save_directory'] + os.sep + 'results.pkl'
        pkl.dump(results, open(results_file_path, 'wb'))

        # save the weight
        if params['save_weights_path']:
            network_params = semi_vae.get_params()
            save_weights(network_params, params['save_weights_path'])

        print('')

params = init_configurations()
train(params)
