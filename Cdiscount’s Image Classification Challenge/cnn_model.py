from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

# slim API has been used for CNN
def CNN(inputs, feature_maps, conv_stride, maxp_stride, is_training=True, x_size=40):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.001)):
        x = tf.reshape(inputs, [-1, x_size, x_size, 3])

        # For slim API, by default the activation function is rectified linear unit "activation_fn=nn.relu"

        net = slim.conv2d(x, feature_maps, conv_stride, scope='conv1-1')
        net = slim.max_pool2d(net, maxp_stride, scope='pool1')
        net = slim.conv2d(net, 2 * feature_maps, conv_stride, scope='conv2-1')
        net = slim.max_pool2d(net, maxp_stride, scope='pool2')
        net = slim.conv2d(net, 2 * 2 * feature_maps, conv_stride, scope='conv3-1')
        net = slim.conv2d(net, 2 * 2 * feature_maps, conv_stride, scope='conv4-1')
        net = slim.max_pool2d(net, maxp_stride, scope='pool3')
        net = slim.flatten(net, scope='flatten1')
        net = slim.fully_connected(net, 2048, scope='fc1')
        net = slim.dropout(net, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, is_training=is_training, scope='dropout2')  # 0.5 by default
        outputs = slim.fully_connected(net, 5, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs
