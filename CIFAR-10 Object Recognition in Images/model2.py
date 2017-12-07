# OSMAN FURKAN KINLI - S002969 - Computer Science in Engineering @ Ozyegin University
# Assignment-2 in CS456 Introduction to Deep Learning Course

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import read_data

FILEPATH = '' # File path should be given here.
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size2', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir2', FILEPATH + '/cifar10_data', # please change this to current data path!!
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp162', False,
                            """Train the model using fp16.""")

# Global constants
IMAGE_SIZE = read_data.IMAGE_SIZE
NUM_CLASSES = read_data.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = read_data.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = read_data.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        type = tf.float16 if FLAGS.use_fp162 else tf.float32
        variable = tf.get_variable(name, shape, initializer=initializer, dtype=type)
    return variable


def _variable_with_weight_decay(name, shape, stddev, wd):
    type = tf.float16 if FLAGS.use_fp162 else tf.float32
    variable = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=type))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(variable), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return variable


def inputs(eval_data):
    if not FLAGS.data_dir2:
        raise ValueError('Data could not been found!')
    data_dir = os.path.join(FLAGS.data_dir2, 'cifar-10-batches-bin')
    images, labels = read_data.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size2)
    if FLAGS.use_fp162:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def create_model(images):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 3, 32],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 32],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 32],
                                             stddev=5e-2,
                                             wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, 32, 32],
                                                 stddev=5e-2,
                                                 wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4)

    norm2 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')

    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size2, -1])
        dim = reshape.get_shape()[1].value
        fc1_weights = _variable_with_weight_decay('weights', shape=[dim, 216],
                                                  stddev=0.04, wd=0.004)
        fc1_biases = _variable_on_cpu('biases', [216], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases, name=scope.name)
        _activation_summary(local3)

    with tf.variable_scope('local4') as scope:
        fc2_weights = _variable_with_weight_decay('weights', shape=[216, 108],
                                                  stddev=0.04, wd=0.004)
        fc2_biases = _variable_on_cpu('biases', [108], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, fc2_weights) + fc2_biases, name=scope.name)
        _activation_summary(local4)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [108, NUM_CLASSES],
                                              stddev=1 / 108.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear, fc1_weights, fc1_biases, fc2_weights, fc2_biases


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size2
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_dir = FLAGS.data_dir2
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = DATA_URL.split('/')[-1]
    path = os.path.join(dest_dir, filename)
    if not os.path.exists(path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        path, _ = urllib.request.urlretrieve(DATA_URL, path, _progress)
        print()
        stat_info = os.stat(path)
        print('Successfully downloaded', filename, stat_info.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(path, 'r:gz').extractall(dest_dir)
