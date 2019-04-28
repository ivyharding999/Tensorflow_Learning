
# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np


# In[2]:


FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('lr',1e-4,"""Learning rate""")     ###这里是1e-4,我一直以为是le-4!!!!注意这里只能运行一次


# In[3]:

##定义卷积运算
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

##定义归一化层
def batch_norm(x, n_out, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')
        gamma = tf.Variable(tf.truncated_normal([n_out], stddev=0.1), name='gamma')
        tf.add_to_collection('biases', beta)
        tf.add_to_collection('weights', gamma)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        normed = tf.nn.batch_norm_with_global_normalization(x, batch_mean, batch_var, 
            beta, gamma, 1e-3, scale_after_normalization=True)
    return normed

##定义残差块
def residual_block(x, n_in, n_out, subsample, scope='res_block'):
    with tf.variable_scope(scope):
        if subsample:
            y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')
            shortcut = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='shortcut')
        else:
            y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
            shortcut = tf.identity(x, name='shortcut')
        y = batch_norm(y, n_out, scope='bn_1')
        y = tf.nn.relu(y, name='relu_1')
        y = conv2d(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
        y = batch_norm(y, n_out, scope='bn_2')
        y = y + shortcut
        y = tf.nn.relu(y, name='relu_2')
    return y

##定义残差组
def residual_group(x, n_in, n_out, n, first_subsample, scope='res_group'):
    with tf.variable_scope(scope):
        y = residual_block(x, n_in, n_out, first_subsample, scope='block_1')
        for i in range(n - 1):
            y = residual_block(y, n_out, n_out, False, scope='block_%d' % (i + 2))
    return y

# In[4]:


def ResNet34(x, n_classes,scope='res_net'):
    with tf.variable_scope(scope):
        y = conv2d(x, 1, 64, 7, 1, 'SAME', False, scope='conv_init')
        y = batch_norm(y, 64, scope='bn_init')
        y = tf.nn.relu(y, name='relu_init')
        y = residual_group(y, 64, 64, 3, False, scope='group_1')
        y = residual_group(y, 64, 128, 4, True, scope='group_2')
        y = residual_group(y, 128, 256, 6 ,True, scope='group_3')
        y = residual_group(y, 256, 512, 3 ,True, scope='group_4')
        with tf.name_scope('deconv1') as scope:  
            wt5_2 = tf.Variable(tf.truncated_normal([3, 3, 512, 512])) 
            y = tf.nn.relu(tf.nn.conv2d_transpose(y, wt5_2, [4, 40, 144, 512], [1, 2, 2, 1],padding='SAME'))
        y = conv2d(y, 512, 64, 1, 1, 'SAME', True, scope='conv_last2')
        y = conv2d(y, 64, n_classes, 1, 1, 'SAME', True, scope='conv_last1')
        #y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], padding='VALID', name='avg_pool')
    return y
       
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
    
def accuracy(logits, gt_label, scope='accuracy'):
    with tf.variable_scope(scope):
        dim = logits.get_shape()[3].value
        logits = tf.reshape(logits, [-1, dim])
        pred_label = tf.argmax(logits, 1)
        acc = 1.0 - tf.nn.zero_fraction(
            tf.cast(tf.equal(pred_label, gt_label), tf.int32))
    return acc

# In[6]:


def train(loss,global_step):
    optimizer=tf.train.AdamOptimizer(1e-4)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op

