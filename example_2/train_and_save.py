import tensorflow as tf
import numpy as np
import os, urllib, gzip, sys
from tensorflow.examples.tutorials.mnist import input_data
# import warnings
# warnings.filterwarnings('ignore', '.*do not.*',)

save_dir = sys.argv[1]

mnist = input_data.read_data_sets('/tmp/tensor', one_hot=True)
graph = tf.Graph()
with graph.as_default():
    examples = tf.placeholder(tf.float32, [None, 784], name='in')
    labels = tf.placeholder(tf.float32, [None, 10])
    weights = tf.Variable(tf.random_uniform([784, 10]))
    bias = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.add(tf.matmul(examples, weights), bias)
    estimates = tf.nn.softmax(logits, name='out')
    cross_entropy = -tf.reduce_sum(labels * tf.log(estimates), [1])
    loss = tf.reduce_mean(cross_entropy)
    gdo = tf.train.GradientDescentOptimizer(0.5)
    optimizer = gdo.minimize(loss)

with tf.Session(graph=graph) as session:
    # Execute the operation directly
    tf.initialize_all_variables().run()
    for step in range(100):
        # Fetch next 100 examples and labels
        x, y = mnist.train.next_batch(100)
        # Ignore the result of the optimizer (None)
        _, loss_value = session.run(
            [optimizer, loss],
            feed_dict={examples: x, labels: y})
        print("Loss at step {0}: {1}".format(step, loss_value))
    tf.saved_model.simple_save(session, save_dir,
                               inputs={'in': examples},
                               outputs={'out': estimates})
