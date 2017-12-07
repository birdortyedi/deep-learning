import get_data, cnn_model
import numpy as np
from datetime import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim

MODEL_DIRECTORY = "model/model.ckpt"
LOGS_DIRECTORY = "logs/train"

# Parameters
training_epochs = 50
TRAIN_BATCH_SIZE = 128
display_step = 100
TEST_BATCH_SIZE = 128

def train(categories, X, y, f_maps, conv_stride, maxp_stride, training_epochs):
    batch_size = TRAIN_BATCH_SIZE
    num_labels = 5

    X_train, X_test, y_train, y_test = get_data.split_data(X, y)
    print("Size of data: %s" % str(X_train.shape))
    print("Length of data: %s" % str(len(X_train)))
    print("Size of labels: %s" % str(y_train.shape))
    print("Length of labels: %s" % str(len(y_train)))
    train_size = len(X_train)
    validation_data = X_test
    validation_labels = y_test

    is_training = tf.placeholder(tf.bool, name='MODE')

    # Tensorflow variables should be initialized.
    x = tf.placeholder(tf.float32, [None, 4800])
    y_ = tf.placeholder(tf.float32, [None, num_labels])

    y = cnn_model.CNN(x, feature_maps=f_maps, conv_stride=conv_stride, maxp_stride=maxp_stride)

    with tf.name_scope("LOSS"):
        loss = slim.losses.softmax_cross_entropy(y, y_)

    tf.summary.scalar('loss', loss)

    with tf.name_scope("ADAM"):
        batch = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(1e-4, batch * batch_size, train_size, 0.95, staircase=True)
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
        train_data_ = X_train
        train_labels_ = y_train

        for i in range(total_batch):
            offset = (i * batch_size) % train_size
            batch_xs = train_data_[offset:(offset + batch_size), :]
            batch_ys = train_labels_[offset:(offset + batch_size), :]

            _, train_accuracy, summary = sess.run([train_step, accuracy, merged_summary_op],
                                                  feed_dict={x: batch_xs, y_: batch_ys, is_training: True})

            # Optional
            summary_writer.add_summary(summary, epoch * total_batch + i)

            if i % display_step == 0:
                format_str = '%s: step %d, accuracy = %.3f'
                print(format_str % (datetime.now(), (epoch+1), train_accuracy))

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
    cats, X, y = get_data.read_data()
    train(cats, X, y, f_maps=128, conv_stride= [5,5], maxp_stride =[2,2], training_epochs=training_epochs)


if __name__ == '__main__':
    main()