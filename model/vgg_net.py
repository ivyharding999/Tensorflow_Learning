
# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np


# In[2]:


FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('lr',1e-4,"""Learning rate""")     ###这里是1e-4,我一直以为是le-4!!!!注意这里只能运行一次


# In[3]:


def conv2d( x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
    with tf.variable_scope(scope):
        kernel=tf.Variable(tf.truncated_normal([ k, k, n_in, n_out], stddev=math.sqrt(2/(k*k*n_in))),name='weight')
        tf.add_to_collection('weight',kernel)
        conv=tf.nn.conv2d(x,kernel,[1,s,s,1],padding=p)
        if bias:
            bias=tf.get_variable('bais',[n_out],initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias',bias)
            conv=tf.nn.bias_add(conv,bias)
    return conv


# In[4]:


def inference(image,scope='vgg_net'):
    with tf.variable_scope(scope):
        #input:   160*576*1
        #output:  160*576*32------80*288*32  参数‘SAME’计算只与步长有关，但是如果参数是：‘VALID’，那么就有一个计算公式
        conv1_1 = conv2d(image, 1, 32, 3, 1, 'SAME', True, scope='conv1_1') 
        conv1_relu = tf.nn.relu(conv1_1, name='relu_conv1_1')

        conv1_2 = conv2d(conv1_relu, 32, 32, 3, 1, 'SAME', True, scope='conv1_2')
        conv1_2_relu = tf.nn.relu(conv1_2, name='relu_conv1_2')

        pool1  = tf.nn.max_pool(conv1_2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')

        #input:   80*288*32
        #output:  80*288*64------40*144*64
        conv2_1 = conv2d(pool1, 32, 64, 3, 1, 'SAME', True, scope='conv2_1')
        conv2_1_relu = tf.nn.relu(conv2_1, name='relu_conv2_1')

        conv2_2 = conv2d(conv2_1_relu, 64, 64, 3, 1, 'SAME', True, scope='conv2_2')
        conv2_2_relu = tf.nn.relu(conv2_2, name='relu_conv2_2')

        pool2 = tf.nn.max_pool(conv2_2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')
       
        #input:   40*144*64
        #output:  40*144*128------20*72*128
        conv3_1 = conv2d(pool2, 64, 128, 3, 1, 'SAME', True, scope='conv3_1')
        conv3_1_relu = tf.nn.relu(conv3_1, name='relu_conv3_1')

        conv3_2 = conv2d(conv3_1_relu, 128, 128, 3, 1, 'SAME', True, scope='conv3_2')
        conv3__2relu = tf.nn.relu(conv3_2, name='relu_conv3_2')

        conv3_3 = conv2d(conv3__2relu, 128, 128, 3, 1, 'SAME', True, scope='conv3')
        conv3_3_relu = tf.nn.relu(conv3_3, name='relu_conv3_3')

        pool3 = tf.nn.max_pool(conv3_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool3')

        #input:   20*72*128
        #output:  20*72*256------10*36*256
        conv4_1 = conv2d(pool3, 128, 256, 3, 1, 'SAME', True, scope='conv4_1')
        conv4_1_relu = tf.nn.relu(conv4_1, name='relu_conv4_1')

        conv4_2 = conv2d(conv4_1_relu, 256, 256, 3, 1, 'SAME', True, scope='conv4_2')
        conv4_2_relu = tf.nn.relu(conv4_2, name='relu_conv4_2')

        conv4_3 = conv2d(conv4_2_relu, 256, 256, 3, 1, 'SAME', True, scope='conv4_3')
        conv4_3_relu = tf.nn.relu(conv4_3, name='relu_conv4_3')

        pool4 = tf.nn.max_pool(conv4_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool4')


        #input:   10*36*256
        #output:  10*36*256------5*18*256
        conv5_1 = conv2d(pool4, 256, 256, 3, 1, 'SAME', True, scope='conv5_1')
        conv5_1_relu = tf.nn.relu(conv5_1, name='relu_conv5_1')

        conv5_2 = conv2d(conv5_1_relu, 256, 256, 3, 1, 'SAME', True, scope='conv5_2')
        conv5_2_relu = tf.nn.relu(conv5_2, name='relu_conv5_2')

        conv5_3 = conv2d(conv5_2_relu, 256, 256, 3, 1, 'SAME', True, scope='conv5_3')
        conv5_3_relu = tf.nn.relu(conv5_3, name='relu_conv5_3')
        pool5 = tf.nn.max_pool(conv5_3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool5')
        #input:   5*18*256
        #output:  10*36*256
        with tf.name_scope('deconv5_1_infrareds') as scope:  
            wt5_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 256])) 
            deconv5_1_infrareds = tf.nn.relu(tf.nn.conv2d_transpose(pool5, wt5_1, 
                                [4, 10, 36, 256], [1, 2, 2, 1],
                                padding='SAME'))
        #input:   10*36*256
        #output:  20*72*256
        with tf.name_scope('deconv5_2_infrareds') as scope:  
            wt5_2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256])) 
            deconv5_2_infrareds = tf.nn.relu(tf.nn.conv2d_transpose(deconv5_1_infrareds, wt5_2, 
                                [4, 20, 72, 256], [1, 2, 2, 1],
                                padding='SAME'))
        #input:   20*72*256
        #output:  40*144*256
        with tf.name_scope('deconv5_3_infrareds') as scope:  
            wt5_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256])) 
            deconv5_3_infrareds = tf.nn.relu(tf.nn.conv2d_transpose(deconv5_2_infrareds, wt5_3, 
                                [4, 40, 144, 256], [1, 2, 2, 1],
                                padding='SAME'))
        #input:   40*144*256
        #output:  40*144*256
        fc4 = conv2d(deconv5_3_infrareds, 256, 256, 1, 1, 'SAME', True, scope='fc4')
        fc4_relu = tf.nn.relu(fc4, name='relu_fc4')
        #input:   40*144*256
        #output:  40*144*32
        fc5 = conv2d(fc4_relu, 256, 32, 1, 1, 'SAME', True, scope='fc5')  ###输出竟然没有经过激活函数！！！！

    return fc5      


# In[5]:


def loss(pre_dep,true_dep):
    dim=pre_dep.get_shape()[3].value
    logits=tf.reshape(pre_dep,[-1,dim])
    labels=tf.reshape(true_dep,[-1])
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy_mean')
    tf.add_to_collection('losses',cross_entropy_mean)
    #weight l2 decay loss  正则项，权重衰减
    weight_l2_losses=[tf.nn.l2_loss(o) for o in tf.get_collection('weight')]
    weight_decay_loss=tf.multiply(1e-4,tf.add_n(weight_l2_losses),name='weight_decay_loss')
    tf.add_to_collection('losses',weight_decay_loss)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')
    


# In[6]:


def train(loss,global_step):
    optimizer=tf.train.AdamOptimizer(1e-4)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op

