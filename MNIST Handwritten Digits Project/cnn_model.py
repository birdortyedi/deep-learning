from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# slim API has been used for CNN
def CNN(inputs, feature_maps, conv_stride, maxp_stride, is_training=True, use_downsampling=True, x_size=28):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, x_size, x_size, 1])

        # For slim API, by default the activation function is rectified linear unit "activation_fn=nn.relu"

        if use_downsampling:
            net = slim.avg_pool2d(x, kernel_size=2, stride=2, scope='resize')
            net = slim.conv2d(net, feature_maps, conv_stride, scope='conv1')
        else:
            net = slim.conv2d(x, feature_maps, conv_stride, scope='conv1')

        net = slim.max_pool2d(net, maxp_stride, scope='pool1')
        net = slim.conv2d(net, 2 * feature_maps, conv_stride, scope='conv2')
        net = slim.max_pool2d(net, maxp_stride, scope='pool2')
        net = slim.flatten(net, scope='flatten3')
        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs
