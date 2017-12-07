#
#   OSMAN FURKAN KINLI
#   S002969 - Computer Science in Engineering
#   Ozyegin University - CS466 Introduction to Deep Learning
#   Assignment-1 - MNIST CNN Multinominal Logistic Regression
#
#   To execute program:
#   "python furkan_kinli_training <network_name> <dataset_name> <path>"
#
#   For network name: network_1 & network_2
#   For dataset name: 28x28_dataset & 14x14_dataset & 14x14_augmented_dataset
#   For path: Please be sure that the path has the MNIST Dataset files, otherwise
#               program will download the MNIST Dataset from Yann LeCun's Website.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys

import tensorflow as tf
import tensorflow.contrib.slim as slim

import mnist_data
import cnn_model

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Parameters
training_epochs = 100
TRAIN_BATCH_SIZE = 50
display_step = 60
TEST_BATCH_SIZE = 100


def train(f_maps, conv_stride, maxp_stride, use_augmentation=True, use_downsampling=True, training_epochs=training_epochs):
    batch_size = TRAIN_BATCH_SIZE
    num_labels = mnist_data.NUM_LABELS

    total_train_data, train_size, validation_data, validation_labels = mnist_data.read_mnist_data(use_augmentation)

    is_training = tf.placeholder(tf.bool, name='MODE')

    # Tensorflow variables should be initialized.
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    y = cnn_model.CNN(x, feature_maps=f_maps, conv_stride=conv_stride, maxp_stride=maxp_stride, use_downsampling=use_downsampling)

    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y, y_)

    tf.summary.scalar('loss', loss)

    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(1e-4,  batch * batch_size, train_size, 0.95, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch)

    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope("ACC"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('acc', accuracy)
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

    total_batch = int(train_size / batch_size)

    # Writing a log file is optional
    summary_writer = tf.summary.FileWriter(LOGS_DIRECTORY, graph=tf.get_default_graph())

    for epoch in range(training_epochs):
        np.random.shuffle(total_train_data)
        train_data_ = total_train_data[:, :-num_labels]
        train_labels_ = total_train_data[:, -num_labels:]

        for i in range(total_batch):
            offset = (i * batch_size) % train_size
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op],
                                                  feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Optional
            summary_writer.add_summary(summary, epoch * total_batch + i)

            if i % display_step == 0:
                print("Epoch:", '%04d,' % (epoch + 1), "batch_index %4d/%4d, Training accuracy: %.5f" % (i, total_batch, train_accuracy))

    saver.save(sess, MODEL_DIRECTORY)

    test_size = validation_labels.shape[0]
    batch_size = TEST_BATCH_SIZE
    total_batch = int(test_size / batch_size)

    saver.restore(sess, MODEL_DIRECTORY)

    acc_buffer = []

    for i in range(total_batch):
        offset = (i * batch_size) % test_size
        batch_xs = validation_data[offset:(offset + batch_size), :]
        batch_ys = validation_labels[offset:(offset + batch_size), :]

        y_final = sess.run(y, feed_dict={x: batch_xs, y_: batch_ys, is_training: False})
        correct_prediction = np.equal(np.argmax(y_final, 1), np.argmax(batch_ys, 1))
        acc_buffer.append(np.sum(correct_prediction) / batch_size)

    print("Test accuracy for this model: %g" % np.mean(acc_buffer))

def main():
    MODEL_DIRECTORY = "model/"
    MODEL1 = "model_for_original_data_with_cnn1.ckpt"
    MODEL2 = "model_for_original_data_with_cnn2.ckpt"
    MODEL3 = "model_for_downsampled_data_with_cnn1.ckpt"
    MODEL4 = "model_for_downsampled_data_with_cnn2.ckpt"
    MODEL5 = "model_for_augmented_data_with_cnn1.ckpt"
    MODEL6 = "model_for_augmented_data_with_cnn2.ckpt"

    if not len(sys.argv) > 1:
        network = 'network_1'
        dataset = '28x28_dataset'
        path = '_'
    else:
        network = sys.argv[1]
        dataset = sys.argv[2]
        path = sys.argv[3]

    if network == 'network_1' and dataset == '28x28_dataset':
        print("28x28 Original Dataset Training with CNN-1 has been starting...")
        MODEL_DIRECTORY += MODEL1
        train(32, [5, 5], [2, 2], use_augmentation=False, use_downsampling=False)
        print("28x28 Original Dataset Training with CNN-1 has been completed.")

    if network == 'network_2' and dataset == '28x28_dataset':
        print("28x28 Original Dataset Training with CNN-2 has been starting...")
        MODEL_DIRECTORY += MODEL2
        train(16, [10,10], [3,3], use_augmentation=False, use_downsampling=False)
        print("28x28 Original Dataset Training with CNN-2 has been completed.")

    if network == 'network_1' and dataset == '14x14_dataset':
        print("14x14 Downsampled Dataset Training with CNN-1 has been starting...")
        MODEL_DIRECTORY += MODEL3
        train(32, [5, 5], [2, 2], use_augmentation=False, use_downsampling=True)
        print("14x14 Downsampled Dataset Training with CNN-1 has been completed.")

    if network == 'network_2' and dataset == '14x14_dataset':
        print("14x14 Downsampled Dataset Training with CNN-2 has been starting...")
        MODEL_DIRECTORY += MODEL4
        train(16, [10,10], [3,3], use_augmentation=False, use_downsampling=True)
        print("14x14 Downsampled Dataset Training with CNN-2 has been completed.")

    if network == 'network_1' and dataset == '14x14_augmented_dataset':
        print("14x14 Augmented Dataset Training with CNN-1 has been starting...")
        MODEL_DIRECTORY += MODEL5
        train(32, [5, 5], [2, 2], use_augmentation=True, use_downsampling=True)
        print("14x14 Augmented Dataset Training with CNN-1 has been completed.")

    if network == 'network_2' and dataset == '14x14_augmented_dataset':
        print("14x14 Augmented Dataset Training with CNN-2 has been starting...")
        MODEL_DIRECTORY += MODEL6
        train(16, [10,10], [3,3], use_augmentation=True, use_downsampling=True)
        print("14x14 Augmented Dataset Training with CNN-2 has been completed.")

if __name__ == '__main__':
    main()
