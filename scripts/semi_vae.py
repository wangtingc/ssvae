'''
Author Wead-Hsu, wead-hsu@github
The implementation for paper tilted with 'semi-supervised
learning with deep generative methods'.
'''
import numpy as np
from theano import tensor as T
from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from lasagne import init
from lasagne import regularization
from theano.sandbox.rng_mrg import MRG_RandomStreams


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
# there is a doubt wether mean operation is better for encoder
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

        self.concat_yz = layers.ConcatLayer([label_layer, self.sampler], axis=1)

        self.decoder_w = layers.DenseLayer(self.concat_yz,
            num_units = self.num_units_hidden_rnn,
            nonlinearity = nonlinearities.identity
            )
        
        # the sentence and mask used here is different from encoder, 
        # it is not safe to used 'embed_layer' and 'mask_layer'....
        self.decoder = layers.LSTMLayer(self.embed_layer,
            num_units = self.num_units_hidden_rnn,
            hid_init = self.decoder_w,
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


    def get_cost_L(self, inputs, kl_w):
        # use sent_embs not sent_idx
        # make it clear which get_output_for is used
        print('getting_cost_L')

        # inputs must obey the order.
        sent_input, sent_embs, mask_input, label_input = inputs
        #sent_embs = self.embed_layer.get_output_for(sent_input)
        sent_enc = self.sent_encoder.get_output_for([sent_embs, mask_input])
        enc = self.encoder.get_output_for(self.concat_xy.get_output_for([sent_enc, label_input]))
        mu_z = self.encoder_mu.get_output_for(enc)
        log_var_z = self.encoder_log_var.get_output_for(enc)
        z = self.sampler.get_output_for([mu_z, log_var_z])
        
        dec_init = self.concat_yz.get_output_for([label_input, z])
        dec_init = self.decoder_w.get_output_for(dec_init)
        # shift sent_embs
        sent_embs_shifted = T.zeros_like(sent_embs)
        sent_embs_shifted = T.set_subtensor(sent_embs_shifted[1:], sent_embs[:-1])
        dec = self.decoder.get_output_for([sent_embs_shifted, mask_input, dec_init])
        dec = self.decoder_shp.get_output_for(dec)
        pred_prob = self.decoder_x.get_output_for(dec)
        # we do not know the batch_size and seqlen until inputs are given
        pred_prob = pred_prob.reshape([sent_embs.shape[0] * sent_embs.shape[1], -1])
        
        l_x = objectives.categorical_crossentropy(pred_prob, sent_input.flatten())
        l_x = (l_x.reshape([sent_embs.shape[0], -1]) * mask_input).sum(1)
        l_z = ((mu_z ** 2 + T.exp(log_var_z) - 1 - log_var_z) * 0.5).sum(1)

        cost_L = l_x + l_z * kl_w
        return cost_L


    def get_cost_U(self, inputs, kl_w):
        print('getting_cost_U')
        sent_input, sent_embs, mask_input = inputs
        classifier_enc = self.classifier_helper.get_output_for([sent_embs, mask_input])
        prob_ys_given_x = self.classifier.get_output_for(classifier_enc)

        '''
        label_input_with = []
	for i in xrange(self.num_classes):
                label_input_with.append(self.convert_onehot(T.zeros([image_input.shape[0]], dtype='int64') + i))

        cost_L_with = []
	for i in xrange(self.num_classes):
                cost_L_with.append(self.get_cost_L([image_input, label_input_with[i]]))

        weighted_cost_L = T.zeros([image_input.shape[0],])
        for i in xrange(self.num_classes):
                weighted_cost_L += prob_ys_given_x[:, i] * cost_L_with[i]
        '''

        weighted_cost_L = T.zeros([sent_embs.shape[0],])
        for i in xrange(self.num_classes):
            label_input = T.zeros([sent_embs.shape[0], self.num_classes])
            label_input = T.set_subtensor(label_input[:, i], 1)
            cost_L = self.get_cost_L([sent_input, sent_embs, mask_input, label_input], kl_w)
            weighted_cost_L += prob_ys_given_x[:,i] * cost_L

        entropy_y_given_x = objectives.categorical_crossentropy(prob_ys_given_x, prob_ys_given_x)
        cost_U = weighted_cost_L - entropy_y_given_x

        return cost_U


    def get_cost_C(self, inputs):
        print('getting_cost_C')
        sent_input, sent_embs, mask_input, label_input = inputs
        classifier_enc = self.classifier_helper.get_output_for([sent_embs, mask_input])
        prob_ys_given_x = self.classifier.get_output_for(classifier_enc)
        prob_y_given_x = (prob_ys_given_x * label_input).sum(1)
        cost_C = -T.log(prob_y_given_x)
        return cost_C


    def get_cost_for_label(self, inputs, kl_w):
        cost_L = self.get_cost_L(inputs, kl_w)
        cost_C = self.get_cost_C(inputs)
        return cost_L.mean() + self.beta * cost_C.mean()


    def get_cost_for_unlabel(self, inputs, kl_w):
        cost_U = self.get_cost_U(inputs, kl_w)
        return cost_U.mean()


    def get_cost_together(self, inputs, kl_w):
        sent_l, mask_l, label, sent_u, mask_u = inputs # l for label u for unlabel
        sent_embs_l = self.embed_layer.get_output_for(sent_l)
        sent_embs_u = self.embed_layer.get_output_for(sent_u)

        cost_for_label = self.get_cost_for_label([sent_l, sent_embs_l, mask_l, label], kl_w) * sent_l.shape[0]
        cost_for_unlabel = self.get_cost_for_unlabel([sent_u, sent_embs_u, mask_u], kl_w) * sent_u.shape[0]
        cost_together = (cost_for_label + cost_for_unlabel) / (sent_l.shape[0] + sent_u.shape[0])
        cost_together += self.get_cost_prior() * self.decay_rate
        return cost_together


    def get_cost_test(self, inputs):
        sent_input, mask_input, label_input = inputs
        sent_embs = self.embed_layer.get_output_for(sent_input)
        classifier_enc = self.classifier_helper.get_output_for([sent_embs, mask_input])
        prob_ys_given_x = self.classifier.get_output_for(classifier_enc)
        #cost_test = objectives.categorical_crossentropy(prob_ys_given_x, label_input)
        cost_acc = T.eq(T.argmax(prob_ys_given_x, axis=1), T.argmax(label_input, axis=1))

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
        params += self.decoder.get_params()
        params += self.decoder_x.get_params()
        params += self.classifier_helper.get_params()
        params += self.classifier.get_params()
        params += self.sampler.get_params()

        return params
