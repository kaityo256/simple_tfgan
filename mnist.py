# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# This file was modified by H. Watanabe
# The original file can be found at the follwoing URL.
# https://github.com/tensorflow/models
# ==============================================================================
import gzip
import sys
import os
import urllib
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_FILE = 'mnist.tfrecord'

_IMAGE_SIZE = 28
_NUM_CHANNELS = 1


def image_to_tfexample(image_data, image_format):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format)
    }))


def _extract_images(filename, num_images):
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE,
                            _IMAGE_SIZE, _NUM_CHANNELS)
    return data


def _add_to_tfrecord(data_filename, num_images, tfrecord_writer):
    images = _extract_images(data_filename, num_images)

    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_png(image)

        with tf.Session('') as sess:
            for j in range(num_images):
                sys.stdout.write('\r>> Converting image %d/%d' %
                                 (j + 1, num_images))
                sys.stdout.flush()

                png_string = sess.run(
                    encoded_png, feed_dict={image: images[j]})

                example = image_to_tfexample(png_string, 'png'.encode())
                tfrecord_writer.write(example.SerializeToString())


def _download_dataset():
    filename = _TRAIN_DATA_FILENAME
    if tf.gfile.Exists(filename):
        print("MNIST Dataset alread exists.")
        return
    print('Downloading file %s...' % filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    urllib.request.urlretrieve(_DATA_URL + filename, filename, _progress)
    print()
    with tf.gfile.GFile(filename) as f:
        size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')


def run():
    if tf.gfile.Exists(_TRAIN_FILE):
        print('A TFRecord file already exists.')
        return

    _download_dataset()

    with tf.python_io.TFRecordWriter(_TRAIN_FILE) as tfrecord_writer:
        _add_to_tfrecord(_TRAIN_DATA_FILENAME, 60000, tfrecord_writer)

    print('\nFinished converting the MNIST dataset!')


if __name__ == '__main__':
    run()
