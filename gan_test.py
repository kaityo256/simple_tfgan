# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# This file was modified by H. Watanabe
# The original file can be found at the follwoing URL.
# https://github.com/tensorflow/models
# ==============================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
finally:
    import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tfgan = tf.contrib.gan
layers = tf.contrib.layers
framework = tf.contrib.framework
slim = tf.contrib.slim

#TRAIN_DATA = 'mnist_train.tfrecord'
TRAIN_DATA = 'hiragana.tfrecord'
TOTAL_STEPS = 1600
INTERVAL = 25
_NUM_CLASSES = 10
BATCH_SIZE = 32


def get_split(source):
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    }
    datanum = sum(1 for _ in tf.python_io.tf_record_iterator(source))
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[28, 28, 1], channels=1),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    return slim.dataset.Dataset(
        data_sources=source,
        reader=reader,
        decoder=decoder,
        num_samples=datanum,
        num_classes=_NUM_CLASSES,
        items_to_descriptions=None)


def leaky_relu(net):
    return tf.nn.leaky_relu(net, alpha=0.01)


def visualize_training_generator(data_np, filename):
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')
    print(filename)
    plt.savefig(filename)


def provide_data(source, batch_size):
    dataset = get_split(source)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=1,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size,
        shuffle=True)
    image, = provider.get(['image'])
    image = (tf.cast(image, tf.float32) - 128.0) / 128.0
    images = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=1,
        capacity=5*batch_size)
    return images


def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    f1 = framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu,
        normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay))
    f2 = framework.arg_scope(
        [layers.batch_norm],
        is_training=is_training,
        zero_debias_moving_mean=True)
    with f1, f2:
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        net = layers.conv2d(net, 1, 4, normalizer_fn=None,
                            activation_fn=tf.tanh)
        return net


def discriminator_fn(img, _, weight_decay=2.5e-5,
                     is_training=True):
    with framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu,
            normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(
                net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)


if not tf.gfile.Exists(TRAIN_DATA):
    print("Could not find datasets. Run prepare_data.py.")
    exit()

tf.reset_default_graph()

with tf.device('/cpu:0'):
    real_images = provide_data(TRAIN_DATA, BATCH_SIZE)

noise_dims = 64
gan_model = tfgan.gan_model(
    generator_fn,
    discriminator_fn,
    real_data=real_images,
    generator_inputs=tf.random_normal([BATCH_SIZE, noise_dims]))

improved_wgan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=1.0)

generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer)

num_images_to_eval = 500

with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(
        tf.random_normal([num_images_to_eval, noise_dims]),
        is_training=False)

generated_data_to_visualize = tfgan.eval.image_reshaper(
    eval_images[:20, ...], num_cols=10)

train_step_fn = tfgan.get_sequential_train_steps()
global_step = tf.train.get_or_create_global_step()

with tf.train.SingularMonitoredSession() as sess:
    for i in range(TOTAL_STEPS):
        train_step_fn(sess, gan_train_ops, global_step, train_step_kwargs={})
        if i % INTERVAL == 0:
            digits_np = sess.run([generated_data_to_visualize])
            filename = "gen%03d.png" % (i//INTERVAL)
            visualize_training_generator(digits_np, filename)
