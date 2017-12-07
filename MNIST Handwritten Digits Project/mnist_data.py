from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np
from scipy import ndimage

import urllib.request as u
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data/"

# Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 1000

# MNIST Dataset could be downloaded from source if required
def maybe_download(filename):
    if not tf.gfile.Exists(DATA_DIRECTORY):
        tf.gfile.MakeDirs(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = u.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('The MNIST Dataset has been successfully downloaded!', filename, size, 'bytes.')
    return filepath

# Reading the MNIST Dataset images from ubyte format
def read_mnist_images(filename, images_size):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * images_size * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(images_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [images_size, -1])
    return data

# Reading the MNIST Dataset labels from ubyte format
def read_mnist_labels(filename, images_size):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * images_size)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

# Augmenting the MNIST Dataset if required
def augment_train_data(images, labels):
    augmented_img = []
    augmented_lab = []
    count = 0

    for x, y in zip(images, labels):
        count = count+1
        if count%100==0:
            print ('expanding data : %03d / %03d' % (count,np.size(images,0)))

        bg_value = np.median(x)
        image = np.reshape(x, (-1, 28))

        # Rotating the images with 15 degrees with the mean of original data pixels as background value
        angle = np.random.randint(-15,15,1)
        new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

        # Shifting the images with 2 unit distance with the mean of original data pixels as background value
        shift = np.random.randint(-2, 2, 2)
        new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

        augmented_img.append(np.reshape(new_img_, 784))
        augmented_lab.append(y)

    augmented_total_data = np.concatenate((augmented_img, augmented_lab), axis=1)
    np.random.shuffle(augmented_total_data)
    return augmented_total_data

def read_mnist_data(use_data_augmentation=True):
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    test_data = read_mnist_images(test_data_filename, 10000)
    test_labels = read_mnist_labels(test_labels_filename, 10000)

    validation_data = test_data[:VALIDATION_SIZE, :]
    validation_labels = test_labels[:VALIDATION_SIZE,:]
    train_data = test_data[VALIDATION_SIZE:, :]
    train_labels = test_labels[VALIDATION_SIZE:,:]

    if use_data_augmentation:
        total_train_data = augment_train_data(train_data, train_labels)
    else:
        total_train_data = np.concatenate((train_data, train_labels), axis=1)

    train_size = total_train_data.shape[0]

    print('Size of training dataset: %s' % str(train_size))
    print('Size of test dataset: %s' % str(VALIDATION_SIZE))
    return total_train_data, train_size, validation_data, validation_labels

