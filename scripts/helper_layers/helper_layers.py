from theano import tensor as T
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from theano.sandbox.rng_mrg import MRG_RandomStreams

'''
class NormDenseLayer(layers.Layer):
    def __init__(self, incoming, num_units, 
            W = init.Normal(1e-2),  
            nonlinearity = nonlinearities.sigmoid
            ):

        super(NormDenseLayer, self).__init__(incoming)

        self.num_units = num_units
        self.denselayer = layers.DenseLayer(incoming, 
            num_units = num_units, W = W,
            b = init.Constant(0.),
            nonlinearity = nonlinearities.identity
            )
        self.batchnormlayer = layers.BatchNormLayer(self.denselayer)
        self.nonlinearlayer = layers.NonlinearityLayer(self.batchnormlayer, nonlinearity)

    
    def get_output_for(self, input, deterministic = False):
        output = self.denselayer.get_output_for(input)
        output = self.batchnormlayer.get_output_for(output, deterministic)
        output = self.nonlinearlayer.get_output_for(output)

        return output


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


    def get_params(self):
        params = []
        params += self.denselayer.get_params()
        params += self.batchnormlayer.get_params(trainable=True)
        params += self.nonlinearlayer.get_params()

        return params
'''


'''
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
        self.input_h1_layer = NormDenseLayer(incoming,
                num_units =  num_units_hidden,
                W = W,
                nonlinearity = nonlinearity_hidden
                )

        #self.h1_h2_layer = layers.DenseLayer(self.input_h1_layer,
                #num_units = num_units_hidden,
                #W = W,
                #nonlinearity = nonlinearity_hidden
                #)

        #self.h2_output_layer = layers.DenseLayer(self.h1_h2_layer, num_units_output,
                                                 #nonlinearity = nonlinearity_output)


    def get_output_for(self, input, **kwargs):
        h1_activation = self.input_h1_layer.get_output_for(input)
        #h2_activation = self.h1_h2_layer.get_output_for(h1_activation)
        #output_activation = self.h2_output_layer.get_output_for(h2_activation)
        #return output_activation
        return h1_activation


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units_hidden)


    def get_params(self):
        params = []
        params += self.input_h1_layer.get_params()
        #params += self.h1_h2_layer.get_params()
        #params += self.h2_output_layer.get_params()
        return params
'''


class SamplerLayer(layers.MergeLayer):
    def __init__(self, incomings):
        super(SamplerLayer, self).__init__(incomings)
        self.mrg_srng = MRG_RandomStreams()
        self.dim_sampling = self.input_shapes[0][1]
        print('dim_sampling: ', self.dim_sampling)
        return


    def get_output_for(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        self.eps = self.mrg_srng.normal((inputs[0].shape[0], self.dim_sampling))
        return inputs[0] + T.exp(0.5 * inputs[1]) * self.eps


    def get_output_shape_for(self, input_shapes):
        print('samplerlayer shape: ', input_shapes[0])
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]



class MeanLayer(layers.MergeLayer):
    def __init__(self, incoming,
                 mask_input = None,
                 ):
        '''
        params:
            incomings: input layers, [sent_input, mask_input]
            num_units: num_units for inner lstm layer
        '''
        incomings = [incoming]
        if mask_input:
            incomings.append(mask_input)

        super(MeanLayer, self).__init__(incomings)

    def get_output_for(self, inputs, **kwargs):
        sent_input, mask_input =  inputs
        mean = (sent_input * mask_input[:, :, None]).sum(axis = 1)
        mean = mean / mask_input.sum(axis=1)[:, None]
        return mean


    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])
