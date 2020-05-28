
"""Pooling layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
#from .. import backend as K
from keras.engine.topology import Layer

#from .input_layer import InputLayer
#from ..engine.base_layer import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces
from keras.layers.pooling import _Pooling2D

"""
class _Pooling2D(Layer):
#    Abstract class for different pooling 2D layers.
    

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(_Pooling2D, self).__init__(**kwargs)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        raise NotImplementedError

    def call(self, inputs):
        output = self._pooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super(_Pooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


"""

class RMSPooling2D(_Pooling2D):
    """Average pooling operation for spatial data.
    # Arguments
        pool_size: integer or tuple of 2 integers,
            factors by which to downscale (vertical, horizontal).
            (2, 2) will halve the input in both spatial dimension.
            If only one integer is specified, the same window length
            will be used for both dimensions.
        strides: Integer, tuple of 2 integers, or None.
            Strides values.
            If None, it will default to `pool_size`.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, pooled_rows, pooled_cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, pooled_rows, pooled_cols)`
    """
  
    @interfaces.legacy_pooling2d_support
    def __init__(self, pool_size=(0, 0), strides=None, padding='valid', 
                    data_format=None, epsilon=1e-12, **kwargs):
        super(RMSPooling2D, self).__init__(pool_size, strides, padding, 
                                          data_format, **kwargs)
        self.epsilon = epsilon
#        del self.set_mode
    def _pooling_function(self, inputs,pool_size, strides, padding,
                      data_format):
        output = K.pool2d(K.square(inputs), pool_size, strides, 
                          padding, data_format, pool_mode='avg')
        return (K.sqrt(output + self.epsilon))
    
    
"""      
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 data_format=None, **kwargs):
        super(RMSPooling2D, self).__init__(pool_size, strides, padding,
                                               data_format, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        output = K.pool2d(inputs, pool_size, strides,
                          padding, data_format, pool_mode='avg')
        return output
    
"""    
class FeaturePoolLayer(Layer):
    """
lasagne.layers.FeaturePoolLayer(incoming, pool_size, axis=1,
pool_function=theano.tensor.max, **kwargs)
Feature pooling layer
This layer pools across a given axis of the input. By default this is axis
1, which corresponds to the feature axis for :class:`DenseLayer`,
:class:`Conv1DLayer` and :class:`Conv2DLayer`. The layer can be used to
implement maxout.
Parameters
----------
incoming : a :class:`Layer` instance or tuple
    The layer feeding into this layer, or the expected input shape.
pool_size : integer
    the size of the pooling regions, i.e. the number of features / feature
    maps to be pooled together.
axis : integer
    the axis along which to pool. The default value of ``1`` works
    for :class:`DenseLayer`, :class:`Conv1DLayer` and :class:`Conv2DLayer`.
pool_function : callable
    the pooling function to use. This defaults to `theano.tensor.max`
    (i.e. max-pooling) and can be replaced by any other aggregation
    function.
**kwargs
    Any additional keyword arguments are passed to the :class:`Layer`
    superclass.
Notes
-----
This layer requires that the size of the axis along which it pools is a
multiple of the pool size.
    """





    def __init__(self,Input_Shape, pool_size, axis=1, pool_function=K.max,
                 **kwargs):
        super(FeaturePoolLayer, self).__init__(Input_Shape, **kwargs)
        self.pool_size = pool_size
        self.axis = axis
        self.pool_function = pool_function
    
        num_feature_maps = self.input_shape[self.axis]
        if num_feature_maps % self.pool_size != 0:
            raise ValueError("Number of input feature maps (%d) is not a "
                             "multiple of the pool size (pool_size=%d)" %
                             (num_feature_maps, self.pool_size))
    
    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # make a mutable copy
        output_shape[self.axis] = input_shape[self.axis] // self.pool_size
        return tuple(output_shape)
    
    def get_output_for(self, input, **kwargs):
        input_shape = tuple(input.shape)
        num_feature_maps = input_shape[self.axis]
        num_feature_maps_out = num_feature_maps // self.pool_size
    
        pool_shape = (input_shape[:self.axis] +
                      (num_feature_maps_out, self.pool_size) +
                      input_shape[self.axis+1:])
    
        input_reshaped = input.reshape(pool_shape)
        return self.pool_function(input_reshaped, axis=self.axis + 1)
