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


def load_configurations(condig_path):
    params = {}
    with open(config_path, 'r') as f:
        for l in f:
            k = l.strip().split(':')[0].strip()
            v = l.strip().split(':')[1].strip()
            params[k] = v
    print params
    return params
            


def load_dict(params):
    print('loading dict from ', params['dict_path'])
    with gzip.open(params['dict_path']) as f:
        wdict = pkl.load(f)
        f.close()

    return wdict


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

    return semi_vae


def get_f_init(semi_vae, params):
    z = T.matrix()
    hid_init = semi_vae.decoder_w.get_output_for(z)
    f_init = theano.function([z], [hid_init], name='func_init')
    return f_init

def get_f_dec_step(semi_vae, word_init_stage, params):
    x = T.lvector() # n* 1

    da_prev = T.matrix()
    cell_prev = T.matrix()
    hid_prev = T.matrix()
    
    if word_init_stage:
        emb = T.zeros((x.shape[0], int(params['dim_emb'])), dtype=theano.config.floatX)
    else:
        emb = semi_vae.embed_layer.get_output_for(x)

    cell, hid, da = semi_vae.decoder.one_step(emb, cell_prev, hid_prev, da_prev)
    p = semi_vae.decoder_x.get_output_for(hid)

    inputs = [x, da_prev, cell_prev, hid_prev]
    outputs = [p, cell, hid, da]

    f_dec_step = theano.function(inputs, outputs, name='func_dec_step')
    return f_dec_step

def beam_search(y, z, beam_size, f_init, f_dec_step_init, f_dec_step, max_sent_length, params):
    live_k, dead_k = 1, 0
    sample, sample_score = [], []
    hyp_samples, hyp_score, hyp_hid, hyp_cell = [[]] * live_k, np.zeros(live_k).astype('float32'), [], []
    hid_init, = f_init(z)
    hid_prev = np.tile(hid_init, (live_k, 1))
    cell_prev = np.zeros((live_k, int(params['num_units_hidden_rnn'])), dtype='float32')
    x = -1 * np.ones((live_k,)).astype('int32')

    for ii in range(max_sent_length):
        da = np.tile(y, (cell_prev.shape[0], 1))
        #print x.shape
        #print da.shape
        #print cell_prev.shape
        #print hid_prev.shape
        if ii == 0:
            prob_words, hid, cell, _ = f_dec_step_init(x, da, cell_prev, hid_prev)
        else:
            prob_words, hid, cell, _ = f_dec_step(x, da, cell_prev, hid_prev)
    
        cand_scores = hyp_score[:, None] - np.log(prob_words)
        cand_flat = cand_scores.flatten()
        ranks_flat = cand_flat.argsort()[: (beam_size - dead_k)]

        trans_idx = ranks_flat / np.int64(params['num_words'])
        word_idx = ranks_flat % np.int64(params['num_words'])
        costs = cand_flat[ranks_flat]

        new_hyp_samples, new_hyp_scores, new_hyp_hid, new_hyp_cell = [], np.zeros(beam_size-dead_k).astype(theano.config.floatX), [], []
        for idx, [ti, wi] in enumerate(zip( trans_idx, word_idx)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_hid.append(copy.copy(hid[ti]))
            new_hyp_cell.append(copy.copy(cell[ti]))
    
        new_live_k, hyp_samples, hyp_score, hyp_hid, hyp_cell = 0, [], [], [], []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_score.append(new_hyp_scores[idx])
                hyp_hid.append(new_hyp_hid[idx])
                hyp_cell.append(new_hyp_cell[idx])

        hyp_score = np.array(hyp_score)
        live_k = new_live_k

        if new_live_k < 1:
            break

        if dead_k >= beam_size:
            break

        x, hid_prev, cell_prev = np.array([w[-1] for w in hyp_samples]), np.array(hyp_hid), np.array(hyp_cell)
        x = x.astype('int32')
    
    if live_k > 0:
         for idx in range(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_score[idx])


    return sample, sample_score

def show(sample,sample_score, idict):
    #for i in range(len(sample)):
    for i in range(5):
        print 'score:', sample_score[i]
        #print sample[i]
        for j in sample[i][::-1]:
            print idict[j],



def decode(params):
    wdict = load_dict(params)
    idict = dict([(v, k) for k, v in wdict.items()])
    idict[0] = '<EOS>'
    idict[1] = '<UNK>'
    semi_vae = load_model(params)
    f_init = get_f_init(semi_vae, params)
    f_dec_step_init = get_f_dec_step(semi_vae, True, params)
    f_dec_step = get_f_dec_step(semi_vae, False, params)
    y = np.asarray([[0, 1]], dtype=theano.config.floatX)
    #z = np.zeros((1, int(params['dim_z'])), dtype=theano.config.floatX)
    z = np.random.normal(0, 1, (1, int(params['dim_z']))).astype(theano.config.floatX)
    print('z:', z)
    sample, sample_score = beam_search(y, z, 100, f_init, f_dec_step_init, f_dec_step, 100, params)
    show(sample, sample_score, idict)

config_path = '../../results/imdb-sclstm-5000-87.8/config.log'
params = load_configurations(config_path)
params['data_path'] = '../../data/proc/imdb/imdb_u.pkl.gz' # to be tested
params['dict_path'] = '../../data/proc/imdb/imdb_u.dict.pkl.gz'
params['save_weights_path'] = '../../results/imdb-sclstm-5000-87.8/weights.pkl'
#params['emb_path'] = '../data/proc/imdb_emb_u.pkl.gz'
decode(params)
