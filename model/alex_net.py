
# coding: utf-8
###所有基本跑得程序都没有加入NB层，可以尝试对网络加上归一化，看网络的效果！！！！！！
# In[1]:


import math
import tensorflow as tf
import numpy as np


# In[2]:


FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr',1e-4,"""Learning rate""")     ###这里是1e-4,我一直以为是le-4!!!!注意这里只能运行一次


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


def inference(image,scope='alex_net'):
    with tf.variable_scope(scope):
        conv1=conv2d(image, 1, 96, 11, 1, 'SAME', True, scope='conv1')
        conv1_relu=tf.nn.relu(conv1,name='conv1_relu')
        pool1=tf.nn.max_pool(conv1_relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1')
        
        conv2=conv2d(pool1, 96, 256, 5, 1, 'SAME', True, scope='conv2')
        conv2_relu=tf.nn.relu(conv2,name='conv2_relu')
        pool2=tf.nn.max_pool(conv2_relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2')
        
        conv3=conv2d(pool2, 256, 384, 3, 1, 'SAME', True, scope='conv3')
        conv3_relu=tf.nn.relu(conv3,name='conv3_relu')
        
        conv4=conv2d(conv3_relu, 384, 384, 3, 1, 'SAME', True, scope='conv4')
        conv4_relu=tf.nn.relu(conv4,name='conv4_relu')
        
        conv5=conv2d(conv4_relu, 384, 256, 3, 1, 'SAME', True, scope='conv5')
        conv5_relu=tf.nn.relu(conv5,name='conv5_relu')
        pool5=tf.nn.max_pool(conv5_relu, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool5')
        
        with tf.name_scope('deconv5_image') as scope:
            wt5=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv5_image=tf.nn.relu(tf.nn.conv2d_transpose(pool5,wt5,[4,40,144,256],[1,2,2,1],padding='SAME'))

        fc6=conv2d(deconv5_image,256,256,1,1,'SAME',True,scope='fc6')
        fc6_relu=tf.nn.relu(fc6,name='fc6_relu')
        
        fc7=conv2d(fc6_relu,256,32,1,1,'SAME',True,scope='fc7') ###输出竟然没有经过激活函数！！！！
        fc7_relu=tf.nn.relu(fc7,name='fc7_relu')    ###后来加上的！！！！！！！
    return fc7_relu      


# In[5]:


def loss(pre_dep,true_dep):
    dim=pre_dep.get_shape()[3].value
    logits=tf.reshape(pre_dep,[-1,dim])  
    labels=tf.reshape(true_dep,[-1])
    print('pre_dep.shape:',logits.shape)    # re_dep.shape: (23040, 32)    4*40*144=23040
    print('ground_truth.shape:',labels.shape)    # round_truth.shape: (23040,)
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
    optimizer=tf.train.AdamOptimizer(FLAGS.lr)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op

