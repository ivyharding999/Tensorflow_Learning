
# coding: utf-8
###paper；predicte depth，预测三个任务啥的，这里只实现了一个深度估计，但是吧，我们这里采用的损失函数并不是论文中所讲的尺度不变损失函数，而是单纯的分类任务。
# In[1]:


import math
import tensorflow as tf
import numpy as np


# In[2]:


FLAGS=tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('lr',1e-4,"""Learning rate""")     ###这里是1e-4,我一直以为是le-4!!!!注意这里只能运行一次


# In[3]:

def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    tf.add_to_collection('weights', var)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var 


# In[4]:


def _variable_on_cpu(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


# In[5]:


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


# In[6]:


def fc(scope_name, inputs, shape, bias_shape, wd=0.04,  trainable=True):
    with tf.variable_scope(scope_name) as scope:
        #if reuse is True:
        #    scope.reuse_variables()
        flat = tf.reshape(inputs, [-1, shape[0]])
    
        weights = _variable_with_weight_decay(
            'weights',
            shape,
            stddev=0.01,
            wd=wd,
            trainable=trainable
        )
        biases = _variable_on_cpu('biases', bias_shape, tf.constant_initializer(0.1))
        fc = tf.nn.relu_layer(flat, weights, biases, name=scope.name)
        return fc
'''
def fc( x,  n_out, scope='fc'):
    with tf.variable_scope(scope):
        regularizer = tf.contrib.layers.l2_regularizer(0.1)
        fc = tf.layers.dense( x, n_out, activation=tf.nn.relu, kernel_regularizer=regularizer, name="fc")  
    return fc
'''
# In[7]:


def inference(x, scope='eigen_net'):
    with tf.variable_scope(scope):
        ##scale1____________________________________________________________________gloable coarse_scale network
        #input  160*576*1
        #output 20*72*96
        coarse1 = conv2d(x, 1, 96, 11, 4, 'SAME', True, scope='coarse1')
        coarse1_relu = tf.nn.relu(coarse1, name='relu_coarse1')
        pool1  = tf.nn.max_pool(coarse1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool1')
        #input  20*72*96
        #output 10*36*256
        coarse2 = conv2d(pool1, 96, 256, 5, 1, 'SAME', True, scope='coarse2')
        coarse2_relu = tf.nn.relu(coarse2, name='coarse2_relu')
        pool2 = tf.nn.max_pool(coarse2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name='pool2')

        #input  10*36*256
        #output 10*36*384
        coarse3 = conv2d(pool2, 256, 384, 3, 1, 'SAME', True, scope='coarse3')
        coarse3_relu = tf.nn.relu(coarse3, name='coarse3_relu')

        #input  10*36*384
        #output 10*36*384
        coarse4 = conv2d(coarse3_relu, 384, 384, 3, 1, 'SAME', True, scope='coarse4')
        coarse4_relu = tf.nn.relu(coarse4, name='coarse4_relu')

        #input  10*36*384
        #output 10*36*256
        coarse5 = conv2d(coarse4_relu, 384, 256, 3, 1, 'SAME', True, scope='coarse5')
        coarse5_relu = tf.nn.relu(coarse5, name='coarse5_relu')
        
        #input  10*36*256
        #output 20*72*256
        with tf.name_scope('deconv5_image') as scope:
            wt5=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv5_image=tf.nn.relu(tf.nn.conv2d_transpose(coarse5_relu,wt5,[4,20,72,256],[1,2,2,1],padding='SAME'))
        #input  20*72*256
        #output 40*144*256
        with tf.name_scope('deconv6_image') as scope:
            wt6=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv6_image=tf.nn.relu(tf.nn.conv2d_transpose(deconv5_image,wt6,[4,40,144,256],[1,2,2,1],padding='SAME'))
        
        fc6=conv2d(deconv6_image,256,32,1,1,'SAME',True,scope='fc6')
        coarse6=tf.nn.relu(fc6,name='coarse6')
        
        fc7=conv2d(coarse6,32,1,1,1,'SAME',True,scope='fc7')
        coarse7_output=tf.nn.relu(fc7,name='coarse7_output')
    
        ###scale2___________________________________________________________________local fine_scale network
        #input  160*576*1
        #output 40*144*63
        fine1_conv = conv2d(x, 1, 63, 9, 2, 'SAME', True, scope='fine1')
        fine1_relu = tf.nn.relu(fine1_conv,name='fine1_relu')
        fine1 = tf.nn.max_pool(fine1_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                padding='SAME', name='fine_pool1')
        fine1_dropout = tf.nn.dropout(fine1, 0.8)
        
        #input  40*144*63
        #output 40*144*64
        fine2 = tf.concat([fine1_dropout, coarse7_output],3)
        fine3 = conv2d( fine2, 64, 64, 5, 1, 'SAME', True, scope='fine3')
        fine3_relu = tf.nn.relu(fine3,name='fine3_relu')
        fine3_dropout = tf.nn.dropout(fine3_relu, 0.8)
        fine4 = conv2d(fine3_dropout, 64, 64, 5, 1, 'SAME', True, scope='fine4')
        fine4_relu = tf.nn.relu(fine4,name='fine4_relu')

        ###scale3______________________________________________________________________ Higher Resolution 
        #input  160*576*1
        #output 40*144*96
        scale3_conv1 = conv2d(x, 1, 96, 9, 2, 'SAME', True, scope='scale3_conv1')
        scale3_relu = tf.nn.relu(scale3_conv1,name='scale3_relu')
        
        scale3_pool1 = tf.nn.max_pool(scale3_relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], 
                                      padding='SAME', name='scale3_pool1')
        
        #input  40*144*96
        #output 40*144*64+96=160
        scale3_1 = tf.concat([scale3_pool1, fine4_relu],3)
        #input  40*144*64+96=160
        #output 40*144*64
        scale3_2 = conv2d(scale3_1, 160, 64, 5, 1, 'SAME', True, scope='scale3_2')
        scale3_2_relu = tf.nn.relu(scale3_2,name='scale3_2_relu')
        #input  40*144*64
        #output 40*144*32
        scale3_3 = conv2d(scale3_2_relu, 64, 64, 5, 1, 'SAME', True, scope='scale3_3')
        scale3_3_relu = tf.nn.relu(scale3_3,name='scale3_3_relu')
        scale3_4 = conv2d(scale3_3, 64, 32, 5, 1, 'SAME', True, scope='scale3_4')
        scale3_4_relu = tf.nn.relu(scale3_4,name='scale3_4_relu')
        
    return scale3_4_relu


# In[8]:


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


# In[9]:


def train(loss,global_step):
    optimizer=tf.train.AdamOptimizer(1e-4)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op

