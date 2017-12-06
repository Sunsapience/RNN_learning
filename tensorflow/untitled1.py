# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:30:26 2017

@author: wlgzg
"""

import tensorflow as tf
import numpy as np

# In[]
'''
#   32 is batch_size
#   128 is hidden_dim/state_size
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size)       # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
print(h0.shape)        # (32, 128)
output, h1 = cell.call(inputs, h0) #调用call函数

# call 函数只能前进一步,
#即已知 x_t,h_(t-1),输出 y_t,h_t
#在这种简单情况下(没有权重，没有全连接层)， 输出的 output=h1

print(h1.shape)       #(32, 128)
print(output.shape)   #(32, 128)

a=0.1*np.arange(3200).reshape(32,100)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict={inputs:a}
    output_, h1_=sess.run([output, h1],feed_dict)
    output_=h1_
'''
# In[]
'''
lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
print(lstm_cell_2.state_size)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = lstm_cell_2.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态

output, h1 = lstm_cell_2.call(inputs, h0)
# 已知 x_t,h_(t-1),c_(t-1)输出 y_t,h_t,c_t
# 其中 h1[0]=cell_state,h1[1]=h_state(lstm的输出)
# 在本处 output=h_state=h1[1]
print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
print(output.shape)   #(32, 128)

a=0.1*np.arange(3200).reshape(32,100)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict={inputs:a}
    output_, h1_=sess.run([output, h1],feed_dict)
    #output_=h1_[1]
'''
# In[]
'''
**********************************************
'''
'''
由于call函数每次只能前进一步(隐层从 t-1 到 t),
如果序列长度为10，则需要调动10 次call
引用tf.nn.dynamic_rnn 来处理序列问题(t,t+1,...,t+9)
要求输入数据格式为  (batch_size, time_steps, input_size)
可以直接处理这种格式的数据
'''
'''
********************************************
'''
lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32,10, 100))
h0 = lstm_cell_2.zero_state(32, np.float32)
output, state = tf.nn.dynamic_rnn(lstm_cell_2, inputs, initial_state=h0)
print(output.shape)     # (32, 10, 128)
print(state.h)          # shape=(32, 128)
print(state.c)          # shape=(32, 128)

'''
***********************************************
'''
Y = tf.transpose(output, [1, 0, 2])    #shape is (time_steps,batch_size,input_size) (10,32,128)
Y = tf.reshape(Y, [-1, 128])    #shape is (time_steps*batch_size,input_size) (320,128)
Y = tf.split(Y, 10, 0)      #shape is  time_steps * (batch_size,input_size)
'''
Y最后为 time_step 个矩阵，矩阵 shape是(batch_size,input_size)
每一个矩阵表示的是同一时刻的输出
最终
Y[-1]=state[1]  即最后一个的 Y 等于 h_state
'''

a=0.01*np.arange(32000).reshape(32,10,100)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    feed_dict={inputs:a}
    output_, h1_,y=sess.run([output, state,Y],feed_dict)
    # h1[-1]=y[-1]

# In[]



