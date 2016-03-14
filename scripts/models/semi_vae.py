'''
Author Wead-Hsu, wead-hsu@github
The implementation for paper tilted with 'semi-supervised
learning with deep generative methods'.
'''

import sys
sys.path.append('../../')

import numpy as np
import theano
from theano import tensor as T
from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import init
from lasagne import regularization
from theano.sandbox.rng_mrg import MRG_RandomStreams
from helper_layers.sc_lstm import ScLSTMLayer
from helper_layers.helper_layers import *


__All__ = ['SemiVAE']

def normal(x, mean = 0, sd = 1):
    c = - 0.5 * np.log(2*np.pi)
    return  c - T.log(T.abs_(sd)) - (x - mean) ** 2/ (2 * sd ** 2)


# this class is for getting the hidden output of sentence classification
# . Or used for sentence encoder
# there is a doubt whether mean operation is better for encoder

class SemiVAE():
    def __init__(self,
                 w_emb,
                 num_classes,
                 num_units_hidden_common,
                 dim_z,
                 beta,
                 num_units_hidden_rnn,
                 decay_rate,
                 dropout,
                 word_dropout,
                 ):
        '''
         params:
             num_classes: num of classes for labels
             num_units_hidden_common: num_units_hidden for all BasicLayers.
             dim_z: the dimension of z and num_units_output for encoders BaiscLayer
             beta: reweighted alpha
             num_units_hidden_rnn: num_units_hidden for recurrent layers
             decay_rate: for weight regulazition
             dropout: dropout rate
             word_dropout: word dropout rate
        '''

        self.num_classes = num_classes
        self.num_words = w_emb.shape[0]
        self.dim_emb = w_emb.shape[1]
        self.num_units_hidden_common = num_units_hidden_common
        self.dim_z = dim_z
        self.beta = beta
        self.num_units_hidden_rnn = num_units_hidden_rnn
        self.decay_rate = decay_rate
        self.dropout = dropout
        self.grad_clipping = 0 # to be tested
        self.mrg_srng = MRG_RandomStreams()
        self.word_dropout = word_dropout

        self.build_model(w_emb)


    def build_model(self, w_emb):
            
        self.x_layer = layers.InputLayer((None, None))
        self.m_layer = layers.InputLayer((None, None))
        self.y_layer = layers.InputLayer((None, self.num_classes))

        # =============== embedding ===================
        self.embed_layer = layers.EmbeddingLayer(self.x_layer,
            self.num_words,
            self.dim_emb,
            W = w_emb,
            )
        
        # ================ encoding ===================
        # here we can try LSTMlayer and return the last output
        # or use SentMeanEncoder (to be tested)
        self.sent_encoder = layers.LSTMLayer(self.embed_layer,
            num_units = self.num_units_hidden_rnn,
            mask_input = self.m_layer,
            grad_clipping = self.grad_clipping,
            only_return_final = True,
            )

        self.concat_xy = layers.ConcatLayer(
            [self.sent_encoder, self.y_layer],
            axis = 1,
            )
        
        self.encoder = layers.DenseLayer(self.concat_xy,
            num_units = self.num_units_hidden_common,
            nonlinearity = nonlinearities.softplus
            )

        self.encoder_mu = layers.DenseLayer(
            self.encoder,
            self.dim_z,
            nonlinearity = nonlinearities.identity
            )

        self.encoder_log_var = layers.DenseLayer(
            self.encoder,
            self.dim_z,
            nonlinearity = nonlinearities.identity
            )

        # merge encoder_mu and encoder_log_var to get z.
        self.sampler = SamplerLayer([self.encoder_mu, self.encoder_log_var])
    

        # ================== decoding ===============
        #self.concat_yz = layers.ConcatLayer([label_layer, self.sampler], axis=1)
        #self.init_w = theano.shared(np.random.rand(2, self.dim_emb))
        self.decoder_w = layers.DenseLayer(self.sampler,
            num_units = self.num_units_hidden_rnn,
            nonlinearity = nonlinearities.identity,
            b = init.Constant(0.0),
            )
        
        self.decoder = ScLSTMLayer(self.embed_layer,
            num_units = self.num_units_hidden_rnn,
            hid_init = self.decoder_w,
            da_init = self.y_layer,
            mask_input = self.m_layer,
            grad_clipping = 0, 
            )

        self.decoder_shp = layers.ReshapeLayer(self.decoder, (-1, self.num_units_hidden_rnn))

        # the batch_size and seqlen are not deterministic
        self.decoder_x = layers.DenseLayer(self.decoder_shp,
            num_units = self.num_words,
            nonlinearity = nonlinearities.softmax
            )
        

        # ======================= classifier =====================
        self.classifier = layers.LSTMLayer(self.embed_layer,
            num_units = self.num_units_hidden_rnn,
            mask_input = self.m_layer,
            grad_clipping = 0,
            )

        self.classifier = layers.DropoutLayer(self.classifier,
            p = self.dropout,
            )
            
        self.classifier = MeanLayer(self.classifier,
            mask_input = self.m_layer,
            )

        self.classifier = layers.DenseLayer(
            self.classifier,
            num_units = self.num_classes,
            nonlinearity = nonlinearities.softmax,
            )


    def convert_onehot(self, label_input_cat):
        return T.eye(self.num_classes)[label_input_cat].reshape([label_input_cat.shape[0], -1])


    def forward(self, inputs_enc, inputs_dec, deterministic=False):
        # shift emb if not inversing
        #emb_shifted = T.zeros_like(emb)
        #emb_shifted = T.set_subtensor(emb_shifted[:, 1:, :], emb[:, :-1, :])
        #emb_shifted = T.set_subtensor(emb_shifted[:, 0, :], T.dot(y, self.init_w))

        # inputs must obey the order.
        emb_enc, m_enc, y_enc = inputs_enc
        #emb = self.embed_layer.get_output_for(x)
        mu_z, log_var_z, z = layers.get_output([self.encoder_mu, 
            self.encoder_log_var,
            self.sampler],
            {self.embed_layer: emb_enc, 
            self.m_layer: m_enc,
            self.y_layer: y_enc,
            })
        
        emb_dec, m_dec, y_dec = inputs_dec

        def _word_dropout(emb):
            if self.word_dropout != 0.0:
                m = self.mrg_srng.binomial(emb.shape[:2], 
                                           p=1-self.word_dropout, 
                                           dtype=theano.config.floatX
                                           )
                emb = emb * mask_decoder[:, :, None]
            return emb

        if deterministic:
            emb_dec = _word_dropout(emb_dec)

        print self.sampler, z
        print self.m_layer, m_dec
        print self.y_layer, y_dec
        print self.embed_layer, emb_dec
        
        pred_prob, = layers.get_output([self.decoder_x], 
            {self.embed_layer: emb_dec,
            self.m_layer: m_dec,
            self.y_layer: y_dec,
            self.sampler: z,
            })

        return mu_z, log_var_z, z, pred_prob
 

    def get_cost_L(self, inputs, kl_w, deterministic = False):
        # inputs format should be decided here.
        x, emb, m, y = inputs
        # inverse emb
        inputs_enc = [emb, m, y]


        emb_inverse = emb[:, ::-1, :]
        m_inverse = m[:, ::-1]
        inputs_dec = [emb_inverse, m_inverse, y]
        print emb,emb_inverse
        mu_z, log_var_z, z, pred_prob = self.forward(inputs_enc, inputs_dec, deterministic)
        print pred_prob

        x_inverse = x[:, ::-1]
        x_inverse_shifted = T.zeros_like(x_inverse)
        x_inverse_shifted = T.set_subtensor(x_inverse_shifted[:, :-1], x_inverse[:, 1:])

        pred_prob = pred_prob.reshape([emb.shape[0] * emb.shape[1], -1])
        l_x = objectives.categorical_crossentropy(pred_prob, x_inverse_shifted.flatten())
        l_x = (l_x.reshape([emb.shape[0], -1]) * m_inverse).sum(1)
        l_z = ((mu_z ** 2 + T.exp(log_var_z) - 1 - log_var_z) * 0.5).sum(1)

        cost_L = l_x + l_z * kl_w
        return cost_L


    def get_cost_U(self, inputs, kl_w):
        print('getting_cost_U')
        x_all, emb_all, m_all, x_sub, emb_sub, m_sub = inputs
        prob_ys_given_x, = layers.get_output([self.classifier],
                                            {self.embed_layer: emb_all,
                                            self.m_layer: m_all,
                                            },
                                            deterministic = True,
                                            )
        '''
        y_with = []
	for i in xrange(self.num_classes):
                y_with.append(self.convert_onehot(T.zeros([image_input.shape[0]], dtype='int64') + i))

        cost_L_with = []
	for i in xrange(self.num_classes):
                cost_L_with.append(self.get_cost_L([image_input, y_with[i]]))

        weighted_cost_L = T.zeros([image_input.shape[0],])
        for i in xrange(self.num_classes):
                weighted_cost_L += prob_ys_given_x[:, i] * cost_L_with[i]
        '''

        weighted_cost_L = T.zeros([emb_all.shape[0],])
        for i in xrange(self.num_classes):
            y = T.zeros([emb_all.shape[0], self.num_classes])
            y = T.set_subtensor(y[:, i], 1)
            cost_L = self.get_cost_L([x_sub, emb_sub, m_sub, y], kl_w, True)
            weighted_cost_L += prob_ys_given_x[:,i] * cost_L

        entropy_y_given_x = objectives.categorical_crossentropy(prob_ys_given_x, prob_ys_given_x)
        cost_U = weighted_cost_L - entropy_y_given_x

        # save internal results for debugging
        self.cost_u_Lw = weighted_cost_L
        self.cost_u_E = - entropy_y_given_x

        return cost_U


    def get_cost_C(self, inputs):
        print('getting_cost_C')
        x, emb, m, y = inputs
        prob_ys_given_x, = layers.get_output([self.classifier],
                                            {self.embed_layer: emb,
                                            self.m_layer: m
                                            },
                                            deterministic = False,
                                            )
        prob_y_given_x = (prob_ys_given_x * y).sum(1)
        cost_C = -T.log(prob_y_given_x)
        acc = T.eq(T.argmax(prob_ys_given_x, axis=1), T.argmax(y, axis=1))
        return cost_C, acc


    def get_cost_for_label(self, inputs, kl_w):
        x_all, emb_all, m_all, x_sub, emb_sub, m_sub, y = inputs
        cost_L = self.get_cost_L([x_sub, emb_sub, m_sub, y], kl_w, True)
        cost_C, acc = self.get_cost_C([x_all, emb_all, m_all, y])
        # save internal results
        self.cost_l_L = cost_L
        self.cost_l_C = cost_C
        return cost_L.mean() + self.beta * cost_C.mean(), acc.mean()


    def get_cost_for_unlabel(self, inputs, kl_w):
        cost_U = self.get_cost_U(inputs, kl_w)
        return cost_U.mean()


    def get_cost_together(self, inputs_l, inputs_u, kl_w):
        x_l_all, m_l_all, x_l_sub, m_l_sub, y_l = inputs_l # l for label u for unlabel
        x_u_all, m_u_all, x_u_sub, m_u_sub = inputs_u

        # get emb
        emb_l_all = self.embed_layer.get_output_for(x_l_all)
        emb_l_sub = self.embed_layer.get_output_for(x_l_sub)
        emb_u_all = self.embed_layer.get_output_for(x_u_all)
        emb_u_sub = self.embed_layer.get_output_for(x_u_sub)
        
        inputs_l_with_emb = [x_l_all, emb_l_all, m_l_all, x_l_sub, emb_l_sub, m_l_sub, y_l]
        inputs_u_with_emb = [x_u_all, emb_u_all, m_u_all, x_u_sub, emb_u_sub, m_u_sub]

        cost_l, acc = self.get_cost_for_label(inputs_l_with_emb, kl_w)
        cost_u = self.get_cost_for_unlabel(inputs_u_with_emb, kl_w)
        cost_l = cost_l * x_l_all.shape[0]
        cost_u = cost_u * x_u_all.shape[0]
        cost_together = (cost_l + cost_u) / (x_l_all.shape[0] + x_u_all.shape[0])
        cost_together += self.get_cost_prior() * self.decay_rate

        return cost_together, acc


    def get_cost_test(self, inputs):
        x, m, y = inputs
        emb = self.embed_layer.get_output_for(x)
        prob_ys_given_x, = layers.get_output([self.classifier],
                                            {self.embed_layer: emb,
                                            self.m_layer: m
                                            },
                                            deterministic = False,
                                            )
        #cost = objectives.categorical_crossentropy(prob_ys_given_x, y)
        acc = T.eq(T.argmax(prob_ys_given_x, axis=1), T.argmax(y, axis=1))

        return acc.mean()


    def get_cost_prior(self):
        prior_cost = 0
        params = self.get_params(only_trainable = True)
        for param in params:
            if param.name == 'W':
                prior_cost += regularization.l2(param).sum()

        return prior_cost


    def get_params(self, only_trainable = False):
        params = []
        '''
        params += self.embed_layer.get_params()
        params += self.sent_encoder.get_params()
        params += self.encoder.get_params()
        params += self.encoder_mu.get_params()
        params += self.encoder_log_var.get_params()
        params += self.decoder_w.get_params()
        #params += [self.init_w]
        params += self.decoder.get_params()
        params += self.decoder_x.get_params()
        params += self.classifier_helper.get_params()
        params += self.classifier.get_params()
        params += self.sampler.get_params()
        '''
        if only_trainable:
            params = layers.get_all_params([self.decoder_x, self.classifier], trainable = True)
        else:
            params = layers.get_all_params([self.decoder_x, self.classifier])

        return params
