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
from models.semi_vae import SemiVAE
import copy

def init_configurations():
    params = {}
    params['exp_name'] = 'semi_20k_sc_0.2_5k'
    params['data'] = 'imdb'
    params['data_path'] = '../../data/proc/imdb/imdb_u.pkl.gz' # to be tested
    params['dict_path'] = '../../data/proc/imdb/imdb_u.dict.pkl.gz'
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
    params['save_directory'] = '../../results/semi_vae_' + params['exp_time']
    params['save_weights_path'] = params['save_directory'] + '/weights.pkl'
    params['load_weights_path'] = None
    params['num_seqs'] = None
    params['len_seqs'] = 200
    params['word_dropout'] = 0.0
    params['dropout'] = 0.2
    return params


def load_configurations(config_path):
    #params = pkl.load(open(config_path))
    params = {}
    with open(config_path, 'r') as f:
        for l in f:
            k = l.split(':')[0].strip()
            v = l.split(':')[1].strip()
            params[k] = v
    
    def is_number(x):
        for i in x:
            if '0' > i or i > '9':
                return False
        return True

    for k in params:
        if is_number(params[k]):
            params[k] = int(params[k])
        if params[k] == 'None':
            params[k] = None

    print params
        
    return params
            

def load_model(params):
    #debug
    #params['num_samples_train'] = 22500
    #beta = params['alpha'] * (params['num_samples_train'] + params['num_samples_unlabel']) / params['num_samples_train']
    num_words = int(params['num_words'])
    dim_emb = int(params['dim_emb'])
    dropout = float(params['dropout'])
    num_classes = int(params['num_classes'])
    num_units_hidden_common = int(params['num_units_hidden_common'])
    dim_z = int(params['dim_z'])


    w_emb = np.zeros((num_words, dim_emb), dtype=theano.config.floatX)
    beta = 0
    print('beta', beta)
    semi_vae = SemiVAE(w_emb,
                       int(params['num_classes']),
                       int(params['num_units_hidden_common']),
                       int(params['dim_z']),
                       beta,
                       int(params['num_units_hidden_rnn']),
                       float(params['weight_decay_rate']),
                       float(params['dropout']),
                       float(params['word_dropout']),
                       )

    load_weights(semi_vae.get_params(), params['save_weights_path'])
    
    x = T.imatrix()
    m = T.matrix()
    y = T.matrix()
    mu_z, log_var_z, z = layers.get_output([semi_vae.encoder_mu,
        semi_vae.encoder_log_var,
        semi_vae.sampler],
        {semi_vae.x_layer: x,
        semi_vae.m_layer: m,
        semi_vae.y_layer: y,
        })

    inputs = [x, m, y]
    outputs = [mu_z, log_var_z, z]

    f_forward = theano.function(inputs, outputs)

    return semi_vae, f_forward



config_path = '../../results/imdb-sclstm-5000-87.8/config.log'
params = load_configurations(config_path)
params['data_path'] = '../../data/proc/imdb/imdb_u.pkl.gz' # to be tested
params['dict_path'] = '../../data/proc/imdb/imdb_u.dict.pkl.gz'
params['save_weights_path'] = '../../results/imdb-sclstm-5000-87.8/weights.pkl'
#params['emb_path'] = '../data/proc/imdb_emb_u.pkl.gz'

semi_vae, f_forward = load_model(params)
train, dev, test, unlabel, wdict, w_emb = load_imdb(params)
num_batches = params['num_samples_train'] / params['batch_size']
it = BatchIterator(params['num_samples_train'], params['batch_size'], data = train, testing=True)

x, y, z = [], [], []
num_batches = 10
for batch in xrange(num_batches):
    print '1'
    x_i, y_i = it.next()
    x_sub, m_sub = prepare_data(x_i, None, 400)
    print '2'
    inputs = [x_sub, m_sub, y_i]
    _, _, z_i = f_forward(*inputs)
    print '3'
    z.extend(z_i)
    x.extend(x_i)
    y.extend(y_i)

save_path = 'forward_05k.pkl'
results = {}
results['x'] = x
results['z'] = z
results['y'] = y
with open(save_path, 'wb') as f:
    pkl.dump(results, f)
