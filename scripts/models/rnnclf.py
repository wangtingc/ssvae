import sys
sys.path.append('../')


from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import updates
from helper_layers.helper_layers import MeanLayer
import theano
from theano import tensor as T

class RnnClf:
    def __init__(self, n_words, dim_emb, 
                 num_units,
                 n_classes,
                 w_emb = None,
                 dropout = 0.2,
                 use_final = False,
                 lr = 0.001,
                 pretrain = None,
                 ):
        self.n_words = n_words
        self.dim_emb = dim_emb
        self.num_units = num_units
        self.n_classes = n_classes
        self.lr = lr

        if w_emb is None:
            w_emb = init.GlorotNormal()
    
        self.l_x = layers.InputLayer((None, None))
        self.l_m = layers.InputLayer((None, None))
        self.l_emb = layers.EmbeddingLayer(self.l_x, n_words, dim_emb, W=w_emb)

        if dropout:
            self.l_emb = layers.dropout(self.l_emb, dropout)
        
        if use_final:
            self.l_enc = layers.LSTMLayer(self.l_emb, num_units, mask_input = self.l_m, only_return_final=True)
            self.l_rnn = self.l_enc
        else:
            self.l_enc = layers.LSTMLayer(self.l_emb, num_units, mask_input = self.l_m, only_return_final=False)
            self.l_rnn = self.l_enc
            self.l_enc = MeanLayer(self.l_enc, self.l_m)

        if dropout:
            self.l_enc = layers.dropout(self.l_enc, dropout)

        self.l_y = layers.DenseLayer(self.l_enc, n_classes, nonlinearity=nonlinearities.softmax)


    def load_pretrain(self, pretrain_path):
        import cPickle as pkl
        with open(pretrain_path, 'rb') as f:
            load_wemb = pkl.load(f)
            wemb = self.l_emb.get_params()
            for i in range(len(load_wemb)):
                wemb[i].set_value(load_wemb[i])

            load_lstm = pkl.load(f)
            lstm = self.l_rnn.get_parms()
            for i in range(len(load_lstm)):
                lstm[i].set_value(load_lstm[i])
            


    def get_params(self):
        network_params = layers.get_all_params(self.l_y)
        return network_params
        
    
    def get_f_train(self):
        network_params = self.get_params()
        for param in network_params:
            print param.get_value().shape, param.name
        
        x = T.imatrix()
        m = T.matrix()
        y = T.matrix()
        pred = layers.get_output(self.l_y,
                                {self.l_x: x,
                                self.l_m: m,
                                })

        cost = objectives.categorical_crossentropy(pred, y).mean()
        acc = T.eq(T.argmax(pred, axis=1), T.argmax(y, axis=1)).mean()
        params_update = updates.adam(cost, network_params, self.lr)
        f_train = theano.function([x, m, y], [cost, acc], updates = params_update)
        return f_train


    def get_f_test(self):
        network_params = self.get_params()
        for param in network_params:
            print param.get_value().shape, param.name
        
        x = T.imatrix()
        m = T.matrix()
        y = T.matrix()
        pred = layers.get_output(self.l_y,
                                {self.l_x: x,
                                self.l_m: m,},
                                deterministic=True)

        acc = T.eq(T.argmax(pred, axis=1), T.argmax(y, axis=1)).mean()
        f_test = theano.function([x, m, y], acc)
        return f_test

