from __future__ import print_function
from helpers import merge
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os
import scipy.misc
import tensorflow as tf
from sklearn.cluster import KMeans

slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)

################
# Define flags #
################

flags = tf.app.flags
flags.DEFINE_string("logdir", None, "Directory to save logs")
flags.DEFINE_string("sampledir", None, "Directory to save samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
FLAGS = flags.FLAGS

#########
# Model #
#########


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def generator(z):
    init_width = 7
    filters = (256, 128, 64, 1)
    kernel_size = 4
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        normalizer_fn=slim.batch_norm):
        with tf.variable_scope("gen"):
            net = slim.fully_connected(
                z, init_width ** 2 * filters[0])
            net = tf.reshape(net, [-1, init_width, init_width, filters[0]])
            net = slim.conv2d_transpose(
                net, filters[1],
                kernel_size=kernel_size,
                stride=2)
            net = slim.conv2d_transpose(
                net, filters[2],
                kernel_size=kernel_size,
                stride=1)
            net = slim.conv2d_transpose(
                net,
                filters[3],
                kernel_size=kernel_size,
                stride=2,
                activation_fn=tf.nn.tanh)
            tf.summary.histogram('gen/out', net)
            tf.summary.image("gen", net, max_outputs=8)
    return net


def discriminator(x, reuse, classification=False, dropout=None, int_feats=False):
    filters = (32, 64, )
    kernels = (4, 4)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        reuse=reuse,
                        activation_fn=lrelu):
        with tf.variable_scope("discr"):
            net = tf.reshape(x, [-1, 28, 28, 1])
            net = slim.conv2d(net, filters[0], kernels[0], stride=2, normalizer_fn=slim.batch_norm, scope='conv1')
            net = slim.conv2d(net, filters[1], kernels[1], stride=2, normalizer_fn=slim.batch_norm, scope='conv2')
            if classification:
                with tf.variable_scope("classify"):
                    net = slim.flatten(net,)
                    net = slim.dropout(net, dropout)
                    net = slim.layers.fully_connected(net, 10, activation_fn=None)
            elif int_feats:
                net = slim.flatten(net,)
                return net
            else:
                net = slim.flatten(net,)
                net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='out')
    return net


#############
# Mnist GAN #
#############

def mnist_gan(dataset):
    # Models
    z_dim = 100
    x = tf.placeholder(tf.float32, shape=[None, 784])
    d_model = discriminator(x, reuse=False)

    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    g_model = generator(z)
    dg_model = discriminator(g_model, reuse=True)

    # Optimizers
    t_vars = tf.trainable_variables()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model))
    tf.summary.scalar('d_loss', d_loss)
    d_trainer = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(
        d_loss,
        global_step=global_step,
        var_list=[v for v in t_vars if 'discr/' in v.name])

    g_loss = -tf.reduce_mean(tf.log(dg_model))
    tf.summary.scalar('g_loss', g_loss)
    g_trainer = tf.train.AdamOptimizer(.0002, beta1=.5).minimize(
        g_loss, var_list=[v for v in t_vars if 'gen/' in v.name])

    # Session
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    # Savers
    saver = tf.train.Saver(max_to_keep=20)
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    if checkpoint and not FLAGS.debug:
        print('Restoring from', checkpoint)
        saver.restore(sess, checkpoint)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

    # Training loop
    for step in range(2 if FLAGS.debug else int(1e6)):
        z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
        images = dataset.train.next_batch(FLAGS.batch_size)[0]

        # Update discriminator
        _, d_loss_val = sess.run([d_trainer, d_loss], feed_dict={x: images, z: z_batch})
        # Update generator twice
        sess.run(g_trainer, feed_dict={z: z_batch})
        _, g_loss_val = sess.run([g_trainer, g_loss], feed_dict={z: z_batch})

        # Log details
        print("Gen Loss: ", g_loss_val, " Disc loss: ", d_loss_val)
        summary_str = sess.run(summary, feed_dict={x: images, z: z_batch})
        summary_writer.add_summary(summary_str, global_step.eval())

        # Early stopping
        if np.isnan(g_loss_val) or np.isnan(g_loss_val):
            print('Early stopping')
            break

        if step % 100 == 0:
            # Save samples
            if FLAGS.sampledir:
                samples = 64
                z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
                images = sess.run(g_model, feed_dict={z: z2})
                images = np.reshape(images, [samples, 28, 28])
                images = (images + 1.) / 2.
                scipy.misc.imsave(FLAGS.sampledir + '/sample.png', merge(images, [int(math.sqrt(samples))] * 2))

            # save model
            if not FLAGS.debug:
                checkpoint_file = os.path.join(FLAGS.logdir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=global_step)

    return

##################
# Gan classifier #
##################


def gan_class(dataset):
    # Models
    dropout = .5
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    c_model = discriminator(
        x,
        reuse=False,
        classification=True,
        dropout=keep_prob)

    # Loss
    t_vars = tf.trainable_variables()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(c_model, y))
    optimizer = tf.train.AdamOptimizer(1e-4, beta1=.5)
    trainer = optimizer.minimize(
        loss, var_list=[v for v in t_vars if 'classify/' in v.name])

    # Evaluation metric
    correct_prediction = tf.equal(tf.argmax(c_model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Session
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    # Saver
    restore_vars = {}
    for t_var in t_vars:
        if 'conv' in t_var.name:
            restore_vars[t_var.name.split(':')[0]] = t_var
    saver = tf.train.Saver(restore_vars, max_to_keep=20)
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    if not checkpoint:
        return
    saver.restore(sess, checkpoint)

    # Training loop
    training_size = 400
    for i in range(2000):
        images, labels = dataset.train.next_batch(FLAGS.batch_size)
        if dataset.train._index_in_epoch > training_size:
            dataset.train._index_in_epoch = 0
        # Train
        _, loss_val = sess.run([trainer, loss], feed_dict={x: images, y: labels, keep_prob: dropout})
        if i % 100 == 0:
            print('Loss', loss_val)
        if i % 400 == 0:
            test_accuracy = sess.run(accuracy, feed_dict={
                x: dataset.test.images[:200],
                y: dataset.test.labels[:200],
                keep_prob: 1.})
            print("test accuracy %g" % test_accuracy)
    return


#####################
# KMeans classifier #
#####################

def kmeans(dataset):
    # Models
    x = tf.placeholder(tf.float32, shape=[None, 784])
    feat_model = discriminator(x, reuse=False, int_feats=True)

    # Session
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    # Restore model params
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    saver.restore(sess, checkpoint)

    # Extract intermediate features
    images, labels = dataset.train.next_batch(10000)
    im_features = sess.run(feat_model, feed_dict={x: images})

    # Run kmeans and evaluate
    kmeans = KMeans(n_clusters=10, random_state=0).fit(im_features)
    km_labels = kmeans.labels_
    for i in range(10):
        images_ = images[np.where(km_labels == i)[0]]
        samples = 25
        images_ = np.reshape(images_[:samples], [samples, 28, 28])
        images_ = (images_ + 1.) / 2.
        scipy.misc.imsave('/tmp/cluster%s.png' % i, merge(images_, [int(math.sqrt(samples))] * 2))
    return


##########
# Sample #
##########


def sample():
    if not FLAGS.sampledir:
        print(FLAGS.sampledir, 'is not defined')
        return

    # Model
    z_dim = 100
    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    g_model = generator(z)

    # Session
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    # Restore
    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
    if checkpoint:
        saver.restore(sess, checkpoint)

    # Save samples
    output = FLAGS.sampledir + '/sample.png'
    samples = 64
    z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
    images = sess.run(g_model, feed_dict={z: z2})
    images = np.reshape(images, [samples, 28, 28])
    images = (images + 1.) / 2.
    scipy.misc.imsave(output, merge(images, [int(math.sqrt(samples))] * 2))

########
# Main #
########


def main(_):
    if not tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)
    if FLAGS.sampledir and not tf.gfile.Exists(FLAGS.sampledir):
        tf.gfile.MakeDirs(FLAGS.sampledir)
    if FLAGS.sampledir:
        sample()
        return
    dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
    if FLAGS.classifier:
        gan_class(dataset)
    elif FLAGS.kmeans:
        kmeans(dataset)
    else:
        mnist_gan(dataset)


if __name__ == '__main__':
    tf.app.run()
