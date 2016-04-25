import gzip
import cPickle as pkl
import numpy as np
from misc import *

def load_imdb(params):
    # train: labelled data for training
    # dev: labelled data for dev
    # test: labelled data for testing
    # unlabel: unlabelled data for semi-supervised learning
    params['num_classes'] = 2
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
            assert params['dim_emb'] == w_emb.shape[1]
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
        w_emb = np.random.rand(params['num_words'], params['dim_emb'])
        w_emb = w_emb.astype(theano.config.floatX)

    # filter the dict for all dataset
    assert params['num_words'] <= w_emb.shape[0]
    if not params['emb_path'] or params['num_words'] != w_emb.shape[0]:
        train = [filter_words(train[0], params['num_words']), train[1]]
        dev = [filter_words(dev[0], params['num_words']), dev[1]]
        test = [filter_words(test[0], params['num_words']), test[1]]
        unlabel = [filter_words(unlabel[0], params['num_words'])]
        w_emb = w_emb[: params['num_words'], :]

    return train, dev, test, unlabel, wdict, w_emb



def load_mds(params):
    # train: labelled data for training
    # dev: labelled data for dev
    # test: labelled data for testing
    # unlabel: unlabelled data for semi-supervised learning
    # settings
    params['num_classes'] = 2
    params['num_extra'] = 4
    print('loading data from ', params['data_path'])
    with gzip.open(params['data_path']) as f:
        # origin dataset has no devset
        train = pkl.load(f)
        test = pkl.load(f)
        unlabel = pkl.load(f)
        
        train_of_label = [[[], [], [], []], [[], [], [], []]]
        for i in range(len(train[0])):
            train_of_label[train[1][i][0]][train[1][i][1]].append(train[0][i])

        np.random.seed(1234567890)
        for i in range(params['num_classes']):
            for j in range(params['num_extra']):
                shuffle_idx = np.random.permutation(len(train_of_label[i][i]))
                train_of_label[i][j] = [train_of_label[i][j][k] for k in shuffle_idx]
        
 
        num_of_label = params['num_samples_train'] / (params['num_classes'] * params['num_extra'])
        dev_of_label = [[[], [], [], []], [[], [], [], []]]
        train2unlabel_of_label = [[[], [], [], []], [[], [], [], []]]
        for i in range(params['num_classes']):
            for j in range(params['num_extra']):
                dev_of_label[i][j] = [train_of_label[i][j][k] for k in range(700, 800)]
                train2unlabel_of_label[i][j] = [train_of_label[i][j][k] for k in range(num_of_label, 700)]
                train_of_label[i][j] = [train_of_label[i][j][k] for k in range(num_of_label)]

        
        train_x = []
        train_y = []
        dev_x = []
        dev_y = []
        unlabel_x = unlabel[0]
        unlabel_y = unlabel[1]
        
        for i in range(params['num_classes']):
            for j in range(params['num_extra']):
                train_x.extend(train_of_label[i][j])
                train_y.extend([[i, j]] * len(train_of_label[i][j]))
                dev_x.extend(dev_of_label[i][j])
                dev_y.extend([[i, j]] * len(dev_of_label[i][j]))
                unlabel_x.extend(train2unlabel_of_label[i][j])
                unlabel_y.extend([[-1, j]] * len(train2unlabel_of_label[i][j]))
    
        
        train = [train_x, train_y]
        dev = [dev_x, dev_y]
        unlabel = [unlabel_x, unlabel_y]

        def mds_binarize(y):
            y1 = binarize_labels(np.asarray(y)[:, 0], 2).tolist()
            y2 = binarize_labels(np.asarray(y)[:, 1], 4).tolist()
            return [y1, y2]

        train = [train_x] + mds_binarize(train_y)
        dev = [dev_x] + mds_binarize(dev_y)
        unlabel = [unlabel_x] + mds_binarize(unlabel_y)
        test = [test[0]] + mds_binarize(test[1])

        print len(train[0]), len(train[1])
        print len(dev[0]), len(dev[1])
        print len(test[0]), len(test[1])
        print len(unlabel[0]), len(unlabel[1])

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


    # filter the dict for all dataset
    assert params['num_words'] <= w_emb.shape[0]
    if not params['emb_path'] or params['num_words'] != w_emb.shape[0]:
        train = [filter_words(train[0], params['num_words']), train[1], train[2]]
        dev = [filter_words(dev[0], params['num_words']), dev[1], dev[2]]
        test = [filter_words(test[0], params['num_words']), test[1], test[2]]
        unlabel = [filter_words(unlabel[0], params['num_words']), unlabel[1], unlabel[2]]
        w_emb = w_emb[: params['num_words'], :]
    

    return train, dev, test, unlabel, wdict, w_emb



