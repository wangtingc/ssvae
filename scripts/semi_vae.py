'''
Author Wead-Hsu, wead-hsu@github
The implementation for paper tilted with 'semi-supervised
learning with deep generative methods'.
'''
import numpy as np
import theano
from theano import tensor as T
from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import init
from lasagne import regularization
from theano.sandbox.rng_mrg import MRG_RandomStreams
from sc_lstm import ScLSTMLayer


__All__ = ['SemiVAE']

def normal(x, mean = 0, sd = 1):
    c = - 0.5 * np.log(2*np.pi)
    return  c - T.log(T.abs_(sd)) - (x - mean) ** 2/ (2 * sd ** 2)


class BasicLayer(layers.Layer):
    def __init__(self, incoming,
            num_units_hidden, #num_units_output,
            W = init.Normal(1e-2),
            nonlinearity_hidden = T.nnet.softplus,
            #nonlinearity_output = T.nnet.softplus,
            ):

        super(BasicLayer, self).__init__(incoming)

        self.num_units_hidden = num_units_hidden
        #self.num_units_output = num_units_output

        # the weight and the nonlinearity and W is set locally
        self.input_h1_layer = layers.DenseLayer(incoming,
                num_units =  num_units_hidden,
                W = W,
                nonlinearity = nonlinearity_hidden
                )

        '''
        self.h1_h2_layer = layers.DenseLayer(self.input_h1_layer,
                num_units = num_units_hidden,
                W = W,
                nonlinearity = nonlinearity_hidden
                )
        '''

        #self.h2_output_layer = layers.DenseLayer(self.h1_h2_layer, num_units_output,
                                                 #nonlinearity = nonlinearity_output)


    def get_output_for(self, input):
        h1_activation = self.input_h1_layer.get_output_for(input)
        #h2_activation = self.h1_h2_layer.get_output_for(h1_activation)
        #output_activation = self.h2_output_layer.get_output_for(h2_activation)
        #return output_activation
        return h1_activation


    def get_output_shape_for(self, input_shape):
        return [input_shape[0], self.num_units_hidden]


    def get_params(self):
        params = []
        params += self.input_h1_layer.get_params()
        #params += self.h1_h2_layer.get_params()
        #params += self.h2_output_layer.get_params()
        return params


class SamplerLayer(layers.MergeLayer):
    def __init__(self, incomings):
        super(SamplerLayer, self).__init__(incomings)
        self.mrg_srng = MRG_RandomStreams()
        self.dim_sampling = self.input_shapes[0][1]
        print('dim_sampling: ', self.dim_sampling)
        return


    def get_output_for(self, inputs):
        assert isinstance(inputs, list)
        self.eps = self.mrg_srng.normal((inputs[0].shape[0], self.dim_sampling))
        return inputs[0] + T.exp(0.5 * inputs[1]) * self.eps


    def get_output_shape_for(self, input_shapes):
        print('samplerlayer shape: ', input_shapes[0])
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


    def get_params(self):
        return []



# this class is for getting the hidden output of sentence classification
# . Or used for sentence encoder
# there is a doubt whether mean operation is better for encoder
class SentMeanEncoder(layers.MergeLayer):
    def __init__(self, incomings,
                 num_units,
                 ):
        '''
        params:
            incomings: input layers, [sent_input, mask_input]
            num_units: num_units for inner lstm layer
        '''
        super(SentMeanEncoder, self).__init__(incomings)
        sent_input_layer, mask_input_layer = incomings
        self.num_units = num_units
        self.lstm_layer = layers.LSTMLayer(sent_input_layer,
            num_units = self.num_units,
            mask_input = mask_input_layer,
            grad_clipping = 0 #to be tested
            )


    def get_output_for(self, inputs):
        sent_input, mask_input =  inputs
        lstm_output = self.lstm_layer.get_output_for(inputs)
        lstm_mean = (lstm_output * mask_input[:, :, None]).sum(axis = 1)
        lstm_mean = lstm_mean / mask_input.sum(axis=1)[:, None]
        return lstm_mean


    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], self.num_units)


    def get_params(self):
        params = []
        params += self.lstm_layer.get_params()
        return params


# this class is not lasagne format, since the output is not single and has
# multiple costs functions.
class SemiVAE(layers.MergeLayer):
    def __init__(self,
                 incomings,
                 w_emb,
                 num_units_hidden_common,
                 dim_z,
                 beta,
                 num_units_hidden_rnn,
                 decay_rate,
                 ):
        '''
         params:
             incomings: input layers, [sent_layer, mask_layer, label_layer]
             w_emb: numpy matrix, the word embeddings 
             num_units_hidden_common: num_units_hidden for all BasicLayers.
             dim_z: the dimension of z and num_units_output for encoders BaiscLayer
             beta: reweighted alpha
             num_units_hidden_rnn: num_units_hidden for recurrent layers
        '''

        super(SemiVAE, self).__init__(incomings)
        # random generator
        self.mrg_srng = MRG_RandomStreams()

        self.incomings = incomings
        [sent_layer, mask_layer, label_layer] = self.incomings
        self.num_classes = label_layer.output_shape[1]
        self.num_words = w_emb.shape[0]
        self.dim_emb = w_emb.shape[1]
        self.num_units_hidden_common = num_units_hidden_common
        self.dim_z = dim_z
        self.beta = beta
        self.num_units_hidden_rnn = num_units_hidden_rnn
        self.decay_rate = decay_rate
        self.grad_clipping = 0 # to be tested
            
        # here we can try LSTMlayer and return the last output
        # or use SentMeanEncoder (to be tested)
        self.embed_layer = layers.EmbeddingLayer(sent_layer,
            self.num_words,
            self.dim_emb,
            W = w_emb,
            )
        
        self.sent_encoder = layers.LSTMLayer(self.embed_layer,
            num_units = self.num_units_hidden_rnn,
            mask_input = mask_layer,
            grad_clipping = self.grad_clipping,
            only_return_final = True,
            )

        self.concat_xy = layers.ConcatLayer(
            [self.sent_encoder, label_layer],
            axis = 1,
            )
        
        # actually BasicLayer is just a denselayer
        # wonder if it should be removed for simplicity
        self.encoder = BasicLayer(self.concat_xy,
            num_units_hidden = self.num_units_hidden_common,
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

        #self.concat_yz = layers.ConcatLayer([label_layer, self.sampler], axis=1)
        #self.init_w = theano.shared(np.random.rand(2, self.dim_emb))

        self.decoder_w = layers.DenseLayer(self.sampler,
            num_units = self.num_units_hidden_rnn,
            nonlinearity = nonlinearities.identity,
            b = init.Constant(0.0),
            )
        
        # the sentence and mask used here is different from encoder, 
        # it is not safe to used 'embed_layer' and 'mask_layer'....
        self.decoder = ScLSTMLayer(self.embed_layer,
            num_units = self.num_units_hidden_rnn,
            hid_init = self.decoder_w,
            da_init = label_layer,
            mask_input = mask_layer,
            grad_clipping = 0, 
            )

        self.decoder_shp = layers.ReshapeLayer(self.decoder, (-1, self.num_units_hidden_rnn))

        # the batch_size and seqlen are not deterministic
        self.decoder_x = layers.DenseLayer(self.decoder_shp,
            num_units = self.num_words,
            nonlinearity = nonlinearities.softmax
            )


        self.classifier_helper = SentMeanEncoder(
            [self.embed_layer, mask_layer],
            num_units = self.num_units_hidden_rnn,
            )

        self.classifier = layers.DenseLayer(
            self.classifier_helper,
            num_units = self.num_classes,
            nonlinearity = nonlinearities.softmax,
            )


    def convert_onehot(self, label_input_cat):
        return T.eye(self.num_classes)[label_input_cat].reshape([label_input_cat.shape[0], -1])


    def get_cost_L(self, inputs, kl_w, word_dropout):
        # use sent_embs not sent_idx
        # make it clear which get_output_for is used
        print('getting_cost_L')

        # inputs must obey the order.
        x, embs, m, y = inputs
        #embs = self.embed_layer.get_output_for(x)
        sent_enc = self.sent_encoder.get_output_for([embs, m])
        enc = self.encoder.get_output_for(self.concat_xy.get_output_for([sent_enc, y]))
        mu_z = self.encoder_mu.get_output_for(enc)
        log_var_z = self.encoder_log_var.get_output_for(enc)
        z = self.sampler.get_output_for([mu_z, log_var_z])
        
        #dec_init = self.concat_yz.get_output_for([y, z])
        #dec_init = self.decoder_w.get_output_for(dec_init)
        dec_init = self.decoder_w.get_output_for(z)
        # shift embs
        embs_shifted = T.zeros_like(embs)
        embs_shifted = T.set_subtensor(embs_shifted[:, 1:, :], embs[:, :-1, :])
        #embs_shifted = T.set_subtensor(embs_shifted[:, 0, :], T.dot(y, self.init_w))

        if word_dropout != 0.0:
            mask_decoder = self.mrg_srng.binomial(x.shape[:2], p=1-word_dropout, dtype=theano.config.floatX)
            mask_decoder = mask_decoder[:, :, None]
            embs_shifted = embs_shifted * mask_decoder

        dec = self.decoder.get_output_for([embs_shifted, m, dec_init, y])
        dec = self.decoder_shp.get_output_for(dec)
        pred_prob = self.decoder_x.get_output_for(dec)
        # we do not know the batch_size and seqlen until inputs are given
        pred_prob = pred_prob.reshape([embs.shape[0] * embs.shape[1], -1])
        
        l_x = objectives.categorical_crossentropy(pred_prob, x.flatten())
        l_x = (l_x.reshape([embs.shape[0], -1]) * m).sum(1)
        l_z = ((mu_z ** 2 + T.exp(log_var_z) - 1 - log_var_z) * 0.5).sum(1)

        cost_L = l_x + l_z * kl_w
        return cost_L


    def get_cost_U(self, inputs, kl_w):
        print('getting_cost_U')
        x_all, embs_all, m_all, x_sub, embs_sub, m_sub = inputs
        classifier_enc = self.classifier_helper.get_output_for([embs_all, m_all])
        prob_ys_given_x = self.classifier.get_output_for(classifier_enc)

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

        weighted_cost_L = T.zeros([embs_all.shape[0],])
        for i in xrange(self.num_classes):
            y = T.zeros([embs_all.shape[0], self.num_classes])
            y = T.set_subtensor(y[:, i], 1)
            cost_L = self.get_cost_L([x_sub, embs_sub, m_sub, y], kl_w, 0)
            weighted_cost_L += prob_ys_given_x[:,i] * cost_L

        entropy_y_given_x = objectives.categorical_crossentropy(prob_ys_given_x, prob_ys_given_x)
        cost_U = weighted_cost_L - entropy_y_given_x

        # save internal results for debugging
        self.cost_u_Lw = weighted_cost_L
        self.cost_u_E = - entropy_y_given_x

        return cost_U


    def get_cost_C(self, inputs):
        print('getting_cost_C')
        x, embs, m, y = inputs
        classifier_enc = self.classifier_helper.get_output_for([embs, m])
        prob_ys_given_x = self.classifier.get_output_for(classifier_enc)
        prob_y_given_x = (prob_ys_given_x * y).sum(1)
        cost_C = -T.log(prob_y_given_x)
        return cost_C


    def get_cost_for_label(self, inputs, kl_w, word_dropout):
        x_all, embs_all, m_all, x_sub, embs_sub, m_sub, y = inputs
        cost_L = self.get_cost_L([x_sub, embs_sub, m_sub, y], kl_w, word_dropout)
        cost_C = self.get_cost_C([x_all, embs_all, m_all, y])
        # save internal results
        self.cost_l_L = cost_L
        self.cost_l_C = cost_C
        return cost_L.mean() + self.beta * cost_C.mean()


    def get_cost_for_unlabel(self, inputs, kl_w):
        cost_U = self.get_cost_U(inputs, kl_w)
        return cost_U.mean()


    def get_cost_together(self, inputs_l, inputs_u, kl_w, word_dropout = 0.0):
        x_l_all, m_l_all, x_l_sub, m_l_sub, y_l = inputs_l # l for label u for unlabel
        x_u_all, m_u_all, x_u_sub, m_u_sub = inputs_u

        # get embs
        embs_l_all = self.embed_layer.get_output_for(x_l_all)
        embs_l_sub = self.embed_layer.get_output_for(x_l_sub)
        embs_u_all = self.embed_layer.get_output_for(x_u_all)
        embs_u_sub = self.embed_layer.get_output_for(x_u_sub)
        
        inputs_l_with_emb = [x_l_all, embs_l_all, m_l_all, x_l_sub, embs_l_sub, m_l_sub, y_l]
        inputs_u_with_emb = [x_u_all, embs_u_all, m_u_all, x_u_sub, embs_u_sub, m_u_sub]

        cost_for_label = self.get_cost_for_label(inputs_l_with_emb, kl_w, word_dropout) * x_l_all.shape[0]
        cost_for_unlabel = self.get_cost_for_unlabel(inputs_u_with_emb, kl_w) * x_u_all.shape[0]
        cost_together = (cost_for_label + cost_for_unlabel) / (x_l_all.shape[0] + x_u_all.shape[0])
        cost_together += self.get_cost_prior() * self.decay_rate

        return cost_together


    def get_cost_test(self, inputs):
        x, m, y = inputs
        embs = self.embed_layer.get_output_for(x)
        classifier_enc = self.classifier_helper.get_output_for([embs, m])
        prob_ys_given_x = self.classifier.get_output_for(classifier_enc)
        #cost_test = objectives.categorical_crossentropy(prob_ys_given_x, y)
        cost_acc = T.eq(T.argmax(prob_ys_given_x, axis=1), T.argmax(y, axis=1))

        return cost_acc.mean()


    def get_cost_prior(self):
        prior_cost = 0
        params = self.get_params()
        for param in params:
            if param.name == 'W':
                prior_cost += regularization.l2(param).sum()

        return prior_cost


    def get_params(self):
        params = []
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

        return params
