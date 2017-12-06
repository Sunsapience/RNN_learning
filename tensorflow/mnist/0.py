# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:02:44 2017

@author: wlgzg
"""
# In[1]

import tensorflow as tf
import numpy as np

# In[2]

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In[3]

learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

n_input = 28  
n_steps = 28  
n_hidden = 128  
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
weight= tf.Variable(tf.truncated_normal([n_hidden, n_classes]))
biase = tf.Variable(tf.zeros([n_classes]))

# In[4]

def RNN(_X, weight, biase):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    h_0 = lstm_cell.zero_state(n_hidden, np.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, _X, initial_state=h_0)
    print(outputs)
    logits=tf.matmul(outputs[:,-1,:], weight) + biase
    return logits

# In[5]

pred = RNN(x, weight, biase)
print(pred)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# In[6]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    step=0
    
    while step  < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        if step % 50 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print ("Iter " + str(step) + ", Minibatch Loss= " +          
                "{:.6f}".format(loss) + ", Training Accuracy= " +          
                "{:.5f}".format(acc))
        step+=1
    print ("Optimization Finished!")

# In[9]:

    test_len = batch_size
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))








