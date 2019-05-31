import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mnist

tfgan = tf.contrib.gan
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework
slim = tf.contrib.slim

INPUT_TENSOR = 'inputs:0'
OUTPUT_TENSOR = 'logits:0'
MNIST_DATA_DIR = './'


def leaky_relu(net):
    return tf.nn.leaky_relu(net, alpha=0.01)


def visualize_training_generator(data_np, filename):
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')
    print(filename)
    plt.savefig(filename)


def evaluate_tfgan_loss(gan_loss, name=None):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
    if name:
        print('%s generator loss: %f' % (name, gen_loss_np))
        print('%s discriminator loss: %f' % (name, dis_loss_np))
    else:
        print('Generator loss: %f' % gen_loss_np)
        print('Discriminator loss: %f' % dis_loss_np)


def provide_data(split_name, batch_size, dataset_dir, num_readers=1,
                 num_threads=1):
    dataset = mnist.get_split(split_name, dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size,
        shuffle=(split_name == 'train'))
    [image, label] = provider.get(['image', 'label'])

    # Preprocess the images.
    image = (tf.to_float(image) - 128.0) / 128.0

    # Creates a QueueRunner for the pre-fetching operation.
    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size)

    one_hot_labels = tf.one_hot(labels, dataset.num_classes)
    return images, one_hot_labels, dataset.num_samples


def mnist_score(images, graph_def_filename=None, input_tensor=INPUT_TENSOR,
                output_tensor=OUTPUT_TENSOR, num_batches=1):
    images.shape.assert_is_compatible_with([None, 28, 28, 1])

    graph_def = tfgan.eval.get_graph_def_from_disk(graph_def_filename)

    def mnist_classifier_fn(x): return tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
        x, graph_def, input_tensor, output_tensor)

    score = tfgan.eval.classifier_score(
        images, mnist_classifier_fn, num_batches)
    score.shape.assert_is_compatible_with([])

    return score


def mnist_frechet_distance(real_images, generated_images,
                           graph_def_filename=None, input_tensor=INPUT_TENSOR,
                           output_tensor=OUTPUT_TENSOR, num_batches=1):
    real_images.shape.assert_is_compatible_with([None, 28, 28, 1])
    generated_images.shape.assert_is_compatible_with([None, 28, 28, 1])

    graph_def = tfgan.eval.get_graph_def_from_disk(graph_def_filename)

    def mnist_classifier_fn(x): return tfgan.eval.run_image_classifier(  # pylint: disable=g-long-lambda
        x, graph_def, input_tensor, output_tensor)

    frechet_distance = tfgan.eval.frechet_classifier_distance(
        real_images, generated_images, mnist_classifier_fn, num_batches)
    frechet_distance.shape.assert_is_compatible_with([])

    return frechet_distance


if not tf.gfile.Exists('./mnist_test.tfrecord') or not tf.gfile.Exists('./mnist_train.tfrecord'):
    print("Could not find datasets. Run prepare_data.py")
    exit()

tf.reset_default_graph()

batch_size = 32
with tf.device('/cpu:0'):
    real_images, _, _ = provide_data(
        'train', batch_size, MNIST_DATA_DIR)


def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    with framework.arg_scope([layers.fully_connected, layers.conv2d_transpose], activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm, weights_regularizer=layers.l2_regularizer(weight_decay)),        framework.arg_scope([layers.batch_norm], is_training=is_training, zero_debias_moving_mean=True):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None,
                            activation_fn=tf.tanh)

        return net


def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5,
                     is_training=True):
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(
                net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)


noise_dims = 64
gan_model = tfgan.gan_model(
    generator_fn,
    discriminator_fn,
    real_data=real_images,
    generator_inputs=tf.random_normal([batch_size, noise_dims]))

# Sanity check that generated images before training are garbage.
# check_generated_digits = tfgan.eval.image_reshaper(
#    gan_model.generated_data[:20, ...], num_cols=10)
#visualize_digits(check_generated_digits, "before.png")

# We can use the minimax loss from the original paper.
vanilla_gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.minimax_generator_loss,
    discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)

# We can use the Wasserstein loss (https://arxiv.org/abs/1701.07875) with the
# gradient penalty from the improved Wasserstein loss paper
# (https://arxiv.org/abs/1704.00028).
improved_wgan_loss = tfgan.gan_loss(
    gan_model,
    # We make the loss explicit for demonstration, even though the default is
    # Wasserstein loss.
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=1.0)

# We can also define custom losses to use with the rest of the TFGAN framework.


def silly_custom_generator_loss(gan_model, add_summaries=False):
    return tf.reduce_mean(gan_model.discriminator_gen_outputs)


def silly_custom_discriminator_loss(gan_model, add_summaries=False):
    return (tf.reduce_mean(gan_model.discriminator_gen_outputs) -
            tf.reduce_mean(gan_model.discriminator_real_outputs))


custom_gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=silly_custom_generator_loss,
    discriminator_loss_fn=silly_custom_discriminator_loss)

# Sanity check that we can evaluate our losses.
for gan_loss, name in [(vanilla_gan_loss, 'vanilla loss'),
                       (improved_wgan_loss, 'improved wgan loss'),
                       (custom_gan_loss, 'custom loss')]:
    evaluate_tfgan_loss(gan_loss, name)

generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer)

num_images_to_eval = 500

# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(
        tf.random_normal([num_images_to_eval, noise_dims]),
        is_training=False)

# Calculate Inception score.
eval_score = mnist_score(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

# Calculate Frechet Inception distance.
with tf.device('/cpu:0'):
    real_images, _, _ = provide_data(
        'train', num_images_to_eval, MNIST_DATA_DIR)
frechet_distance = mnist_frechet_distance(
    real_images, eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

# Reshape eval images for viewing.
generated_data_to_visualize = tfgan.eval.image_reshaper(
    eval_images[:20, ...], num_cols=10)

train_step_fn = tfgan.get_sequential_train_steps()

global_step = tf.train.get_or_create_global_step()
loss_values, mnist_scores, frechet_distances = [], [], []

with tf.train.SingularMonitoredSession() as sess:
    for i in range(100):
        cur_loss, _ = train_step_fn(
            sess, gan_train_ops, global_step, train_step_kwargs={})
        loss_values.append((i, cur_loss))
        if i % 10 == 0:
            mnist_score, f_distance, digits_np = sess.run(
                [eval_score, frechet_distance, generated_data_to_visualize])
            filename = "gen%03d.png" % i
            visualize_training_generator(digits_np, filename)
