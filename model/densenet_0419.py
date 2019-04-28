
# coding: utf-8

# In[1]:


import math
import tensorflow as tf
import numpy as np
import os
import scipy.stats
import time
FLAGS=tf.app.flags.FLAGS


# In[2]:


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

def batch_norm(x, n_out, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta')
        gamma = tf.Variable(tf.truncated_normal([n_out], stddev=0.1), name='gamma')
        tf.add_to_collection('biases', beta)
        tf.add_to_collection('weights', gamma)

        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        normed = tf.nn.batch_norm_with_global_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3,
                                                            scale_after_normalization=True)
    return normed

def deconv_layer(x,output_shape, n_in, n_out, k, p='SAME', bias=False, scope='deconv'):
    with tf.variable_scope(scope):
        kernel=tf.Variable(tf.truncated_normal([ k, k, n_out, n_in], stddev=math.sqrt(2/(k*k*n_in))),name='weight')
        tf.add_to_collection('weight',kernel)
        deconv=tf.nn.conv2d_transpose(x,kernel,output_shape,[1,2,2,1],padding=p)
        if bias:
            bias=tf.get_variable('bais',[n_out],initializer=tf.constant_initializer(0.0))
            tf.add_to_collection('bias',bias)
            deconv=tf.nn.bias_add(deconv,bias)
    return deconv


# In[12]:


def bottleneck_layer(x,filters,scope="bottleneck_layer"):
    # print("start bottleneck_layer")
    """
    实现dense_block中获取图像特征的功能
    参数
    x: 输入图像
    filters:卷积后输出feature map的基本个数
    """
    with tf.name_scope(scope):
        x =batch_norm(x, int(x.shape[-1]),scope=scope+"_batch1")
        x = tf.nn.relu(x)
        x = conv2d(x,int(x.shape[-1]), 4*filters, 1, 1,'SAME', False, scope=scope+'_conv1')
        x = tf.nn.dropout(x, 0.8)
    
        x =batch_norm(x, int(x.shape[-1]),scope=scope+"_batch2")
        x = tf.nn.relu(x)
        x = conv2d(x,int(x.shape[-1]), filters, 3, 1,'SAME', False, scope=scope+'_conv2')
        x = tf.nn.dropout(x, 0.8)
        # print(x)
    # print("finish bottleneck_layer--------------------------------")
    return x

def dense_block(input_x,filters,nb_layers,scope):
    # print("start dense_block")
    """
    实现denseblock功能
    参数
    input_x:输入图像
    nb_layers:一个dense_block中包含nb_layers次bottleneck_layer
    filters:卷积后输出feature map的基本个数
    """
    with tf.name_scope(scope):
        layers_concat = list() #创建一个空list
        layers_concat.append(input_x)  #将输入input_x concat到list中
        x = bottleneck_layer(input_x,filters)
        # 将经过bottle_layer中获得的feature map concat到list中
        layers_concat.append(x)   
        for i in range(nb_layers - 1):
            x = tf.concat(layers_concat,axis=3)
            x = bottleneck_layer(x,filters)
            layers_concat.append(x)
        x = tf.concat(layers_concat,axis=3)
    # print("finish dense_block--------------------------------------------------")
    return x
        
def transition_layer(x,k=2,scope='transition_layer'):
    # print("start transition_layer########################")
    """
    实现降采样的功能，使得feature map的尺寸减小
    参数
    x: 输入feature map
    k:avg_pool的kernel参数
    """
    with tf.name_scope(scope):
        x =batch_norm(x, int(x.shape[-1]),scope=scope+"_batch1")
        x = tf.nn.relu(x)
        in_channel = int(x.shape[-1])
        x = conv2d(x,int(x.shape[-1]),int(in_channel*0.5), 1, 1,'SAME', False, scope=scope+'_conv')
        x = tf.nn.dropout(x, 0.8)
        x = tf.nn.avg_pool(x,ksize=[1,k,k,1], strides=[1,k,k,1], padding="VALID")
    return x


# In[ ]:


def UnaryModel( x, filters, scope='unary_model'):
    """
    实现denseNet模型的搭建
    x:输入图像
    filters:卷积后输出feature map的基本个数
    """
    with tf.variable_scope(scope):
        #input  160*576*1
        #output 40*144*32（最多将4倍，否则要做升采样的操作）
        x = conv2d(x,1, 2*filters, 7, 2,'SAME', False, scope='conv') #尺寸变为原来的1/2-
        # print("conv1后尺寸：",x.shape)   #(4, 80, 288, 48)
        # x = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 尺寸变为原来的1/4-
        # print("max_pool后尺寸：",x.shape) # (4, 40, 144, 48)
        
        x = dense_block(input_x=x,filters=filters,nb_layers=6,scope='dense_1')
        # print("finish dense_1----------------------------")
        x = transition_layer(x,k=2)  
        # print("dense_1后尺寸：",x.shape) # dense_1后尺寸： (4, 40, 144, 72)
        
        x = dense_block(input_x=x,filters=filters,nb_layers=12,scope='dense_2')
        x = transition_layer(x,k=2)  # 尺寸变为原来的1/16 
        # print("dense_2后尺寸：",x.shape) # dense_2后尺寸： (4, 20, 72, 192)
        
        x = dense_block(input_x=x,filters=filters,nb_layers=12,scope='dense_3')
        x = transition_layer(x,k=2) 
        # print("dense_3后尺寸：",x.shape) # dense_3后尺寸： (4, 10, 36, 240)

        in_size =  int(x.shape[-1])
        # print("in_size.shape:",in_size)
        
        ## 升采样   input  4*10*36*150    output 4*20*72*150
        with tf.name_scope('deconv1_image') as scope:
            wt1=tf.Variable(tf.truncated_normal([3,3,in_size,in_size]))
            x=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(x,wt1,[4,20,72,in_size],[1,2,2,1],padding='SAME'),in_size))
        #input  4*20*72*256     output 4*40*144*256
        with tf.name_scope('deconv2_image') as scope:
            wt2=tf.Variable(tf.truncated_normal([3,3,in_size,in_size]))
            x=tf.nn.relu(batch_norm(tf.nn.conv2d_transpose(x,wt2,[4,40,144,in_size],[1,2,2,1],padding='SAME'),in_size))
        
        # print("after deconv:",x.shape) # after deconv: (4, 40, 144, in_size)
        x = dense_block(input_x=x,filters=filters,nb_layers=16,scope='dense_4')
        # print("dense_4后尺寸：",x.shape) # dense_4后尺寸： (4, 40, 144, 624)
        
        x = batch_norm(x, 1, scope='convf_bn')
        x=tf.nn.relu(x)
        in_size =  int(x.shape[-1])
        x=conv2d(x,in_size,128,3,1,'SAME',True,scope='fc1')   #[4, 40, 144, 32]     
        x=conv2d(x,128,32,1,1,'SAME',True,scope='fc2')   #[4, 40, 144, 32]     
        print("unary_model has been done. the final size:",x.shape) # (4, 40, 144, 32)  
    return  x

def PairwiseModel( x, scope='pairwise_model'):
    with tf.variable_scope(scope):
        # 160*576*1
        # 40*144*32
        conv1 = conv2d(x, 1, 64, 7, 1, 'SAME', True, scope='conv1')     # 1
        conv1_bn = batch_norm(conv1, 64, scope='conv1_bn')
        conv1_relu = tf.nn.relu(conv1_bn)
        # print("size of conv1_relu: ",conv1_relu.shape) #size of conv1_relu:  (4, 160, 576, 64)
        
        conv2 = conv2d(conv1_relu, 64, 128, 5, 1, 'SAME', True, scope='conv2')   # 2
        conv2_bn = batch_norm(conv2, 128, scope='conv2_bn')
        conv2_relu = tf.nn.relu(conv2_bn)
        # print("size of conv2_relu: ",conv2_relu.shape) # size of conv2_relu:  (4, 160, 576, 128)
        conv1_pooling = tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 尺寸变为原来的1/4-
        conv2_pooling = tf.nn.max_pool(conv2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 尺寸变为原来的1/4-
        x_pooling = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 尺寸变为原来的1/4-
        
        conv2_f = tf.concat([conv1_pooling, conv2_pooling,x_pooling],3)
        # print("size of conv2_f: ",conv2_f.shape) # size of conv2_f:  (4, 80, 288, 193)

        conv3 = conv2d(conv2_f, 193, 256, 3, 1, 'SAME', True, scope='conv3')
        conv3_bn = batch_norm(conv3, 256, scope='conv3_bn')
        conv3_relu = tf.nn.relu(conv3_bn)   
        # print("size of conv3_relu: ",conv3_relu.shape) # size of conv3_relu:  (4, 80, 288, 256)
        conv1_pooling2 = tf.nn.max_pool(conv1_relu,ksize=[1,2,2,1],strides=[1,4,4,1],padding="SAME")# 尺寸变为原来的1/4-
        x_pooling2 = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,4,4,1],padding="SAME")# 尺寸变为原来的1/4-
        conv2_pooling = tf.nn.max_pool(conv2_f,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 尺寸变为原来的1/4-
        conv3_pooling = tf.nn.max_pool(conv3_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 尺寸变为原来的1/4-
        conv3_f = tf.concat([conv1_pooling2, x_pooling2,conv2_pooling,conv3_pooling],3)
        # print("size of conv3_f: ",conv3_f.shape) # size of conv3_f:  (4, 40, 144, 514)
        # conv2_f = tf.concat(conv2_f, conv2_pooling,3)

        conv4 = conv2d(conv3_f, 514, 32, 3, 1, 'SAME', True, scope='conv4')    

        print("pairwise_model has been done...the final size:",conv4.shape)

    return conv4


# In[7]:


def loss(pre_dep,pre_dep_2,true_dep,beta1,beta2,beta3,lamda):
    
    # define unary loss function
    dim=pre_dep.get_shape()[3].value
    logits=tf.reshape(pre_dep,[-1,dim])     # re_dep.shape: (23040, 32)    4*40*144=23040  batch_size*40*144
    labels=tf.reshape(true_dep,[-1])        # ground_truth.shape: (23040,)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy')
    cross_entropy_mean=tf.reduce_mean(cross_entropy,name='cross_entropy_mean')   # 所有元素求平均值
    tf.add_to_collection('losses',cross_entropy_mean)
    print('unary-loss- function-done...')

    # pairwise loss function
    time_start=time.time()
    num = 4      # num 指的是batch_szie
    pred = tf.argmax(pre_dep_2, 3)       # 输出一张depth map(batchsize=4不会变)
    # print('pred.shape------------------------:',pred.shape)   # pred.shape------------------------: (4, 40, 144)
    # print('true_dep.shape------------------------:',true_dep.shape)  # true_dep.shape------------------------: (4, 40, 144, 1)
    feature = tf.reshape(pred,[num,40,144])
    label = tf.reshape(true_dep,[num,40,144])
    # print('feature.dtype:',feature.dtype)   # feature.dtype: <dtype: 'int64'>
    # print('label.dtype:',label.dtype)    # label.dtype: <dtype: 'int64'>
    # int64---->float32
    feature = tf.image.convert_image_dtype(feature,tf.float32)    # 后面的操作只能针对float32进行，而feature是int64
    label = tf.image.convert_image_dtype(label,tf.float32)
    ################## 尺度 1 
    feature_pad = tf.pad(feature ,[[0,0],[1,1],[1,1]],"REFLECT")   # 三个维度：batch_size,height,width.对高和宽分别补（对称补非0操作）
    label_pad = tf.pad(label ,[[0,0],[1,1],[1,1]],"REFLECT")
    feature_new = tf.Variable(tf.zeros([num,40*144,9]))   # 存放特征矩阵,8邻域，针对图片的每个pixel都做了邻域操作，那么一张图片就要做40*144
    label_new = tf.Variable(tf.zeros([num,40*144,9]))     # 存放标签矩阵
    for i in range (1,40+1):
        for j in range (1,144+1):
            index = (i-1)*144 + j-1
            feature_patch = feature_pad[:, i-1:i+2, j-1:j+2] # 取出8邻域元素，有9个
            label_patch = label_pad[:, i-1:i+2, j-1:j+2]
            feature_col = tf.reshape(feature_patch,[-1,9]) 
            label_col = tf.reshape(label_patch,[-1,9])
            feature_new[:,index,:].assign(feature_col)
            label_new[:,index,:].assign(label_col)
    ##########################################  加上尺度2
    feature_mean = tf.reduce_mean(feature_new,2)      # 计算第三个维度的均值  [num,40*144]
    label_mean = tf.reduce_mean(label_new,2)
    feature_new = feature_new[:,:,4:5] - feature_new  # 中心元素减去四周的元素（fi-fj）
    label_new = label_new[:,:,4:5] - label_new        # 中心元素减去四周的元素 （di-dj）
    print('scale_loss1 has been done...')
    # 改变大小--[batch_size,height,width]
    feature_mean = tf.reshape(feature_mean,[num,40,144])
    label_mean = tf.reshape(label_mean,[num,40,144])
    feature_pad_scale2 = tf.pad(feature_mean ,[[0,0],[1,1],[1,1]],"REFLECT")   # 三个维度：batch_size,height,width.对高和宽分别补（对称补非0操作）
    label_pad_scale2 = tf.pad(label_mean ,[[0,0],[1,1],[1,1]],"REFLECT")
    # print ('feature_mean的大小',feature_mean.shape)   # feature_mean的大小 (4, 5760)
    feature_scale2 = tf.Variable(tf.zeros([num,40*144,9]))
    label_scale2 = tf.Variable(tf.zeros([num,40*144,9]))
    for p in range (1,40+1):
        for q in range (1,144+1):
            index_scale2 = (p-1)*144 + q-1
            feature_patch_scale2 = feature_pad_scale2[:, p-1:p+2, q-1:q+2] # 取出8邻域元素，有9个
            label_patch_scale2 = label_pad_scale2[:, p-1:p+2, q-1:q+2]
            feature_col_scale2 = tf.reshape(feature_patch_scale2,[-1,9]) 
            label_col_scale2 = tf.reshape(label_patch_scale2,[-1,9])
            feature_scale2[:,index_scale2,:].assign(feature_col_scale2)
            label_scale2[:,index_scale2,:].assign(label_col_scale2)
    ############################################ 加上尺度3 
    feature_mean_scale3 = tf.reduce_mean(feature_scale2,2)      # 计算第三个维度的均值  [num,40*144]
    label_mean_scale3 = tf.reduce_mean(label_scale2,2)
    feature_scale2 = feature_scale2[:,:,4:5] - feature_scale2  # 中心元素减去四周的元素（fi-fj）
    label_scale2 = label_scale2[:,:,4:5] - label_scale2        # 中心元素减去四周的元素 （di-dj）
    print('scale_loss2 has been done...')
    # 改变大小--[batch_size,height,width]
    feature_mean_scale3 = tf.reshape(feature_mean_scale3,[num,40,144])
    label_mean_scale3 = tf.reshape(label_mean_scale3,[num,40,144])
    feature_pad_scale3 = tf.pad(feature_mean_scale3 ,[[0,0],[1,1],[1,1]],"REFLECT")   # 三个维度：batch_size,height,width.对高和宽分别补（对称补非0操作）
    label_pad_scale3 = tf.pad(label_mean_scale3 ,[[0,0],[1,1],[1,1]],"REFLECT")
    # print ('feature_mean的大小',feature_mean.shape)   # feature_mean的大小 (4, 5760)
    feature_scale3 = tf.Variable(tf.zeros([num,40*144,9]))
    label_scale3 = tf.Variable(tf.zeros([num,40*144,9]))
    for m in range (1,40+1):
        for n in range (1,144+1):
            index_scale3 = (m-1)*144 + n-1
            feature_patch_scale3 = feature_pad_scale3[:, m-1:m+2, n-1:n+2] # 取出8邻域元素，有9个
            label_patch_scale3 = label_pad_scale3[:, m-1:m+2, n-1:n+2]
            feature_col_scale3 = tf.reshape(feature_patch_scale3,[-1,9]) 
            label_col_scale3 = tf.reshape(label_patch_scale3,[-1,9])
            feature_scale3[:,index_scale3,:].assign(feature_col_scale3)
            label_scale3[:,index_scale3,:].assign(label_col_scale3)
    feature_scale3 = feature_scale3[:,:,4:5] - feature_scale3  # 中心元素减去四周的元素（fi-fj）
    label_scale3 = label_scale3[:,:,4:5] - label_scale3        # 中心元素减去四周的元素 （di-dj）
    print('scale_loss3 has been done...')
    ##### 计算loss
    # pairwise_loss.shape = [16,5760,9]，计算了16(batch_size)张图片的损失
    # tf.multiply()----element_wise
    pairwise_loss_sacale1 = lamda*tf.multiply(tf.exp(-beta1* tf.square(feature_new)), tf.square(label_new))  # lamda*exp(-beta*(fi-fj)^2)*(di-dj)^2
    pairwise_loss_sacale2 = lamda*tf.multiply(tf.exp(-beta2* tf.square(feature_scale2)), tf.square(label_scale2))
    pairwise_loss_sacale3 = lamda*tf.multiply(tf.exp(-beta3* tf.square(feature_scale3)), tf.square(label_scale3))
    pairwise_loss = (pairwise_loss_sacale1 + pairwise_loss_sacale2 + pairwise_loss_sacale3)/3
    pairwise_loss = tf.reshape(pairwise_loss,[-1,1])
    pairwise_loss_mean = 9*tf.reduce_mean(pairwise_loss,name='pairwise_loss_mean')  # 所有元素求平均值
    tf.add_to_collection('losses',pairwise_loss_mean)
    print('pairwise-loss- function-done...')
    time_end=time.time()
    print('pairwise time cost',time_end-time_start,'s')   # pairwise time cost 49.375667333602905 s

    weight_l2_losses=[tf.nn.l2_loss(o) for o in tf.get_collection('weight')]
    # print('weight_l2_losses:-------------------',weight_l2_losses[10]) # weight_l2_losses:Tensor("L2Loss_10:0", shape=(), dtype=float32)
    weight_decay_loss=tf.multiply(1e-4,tf.add_n(weight_l2_losses),name='weight_decay_loss')
    tf.add_to_collection('losses',weight_decay_loss)
    print('L2-loss- function-done...')
    return tf.add_n(tf.get_collection('losses'),name='total_loss')



def train_op(loss,global_step):
    optimizer=tf.train.AdamOptimizer(FLAGS.lr)
    train_op=optimizer.minimize(loss,global_step=global_step)
    return train_op


# def accurcy(pre,labels):
#     pred = np.argmax(pre, 3)      # array[4,40,144]  
#     d = np.reshape(labels, (-1))+1    # 拉直
#     d_p = np.reshape(pred, (-1))+1       # 拉直
    
#     rel+= np.sum(np.abs(d - d_p) / d )
#     log+= np.sum(np.abs(np.log10(d) - np.log10(d_p)))
#     rms+= np.sum(np.square((d - d_p)))   # np.square()平方
#     threshold= np.append(threshold, np.maximum(d/d_p, d_p/d))
#     a = threshold < 1.25
#     b = threshold < 1.25*1.25
#     c = threshold < 1.25*1.25*1.25
#     rel = rel/ (4*40*144)
#     log = log/ (4*40*144)
#     rms = np.sqrt(rms/(4*144*40*inter))
#     threshold1 = np.sum(a == True)/(4*144*40)
#     threshold2 = np.sum(b == True)/(4*144*40)
#     threshold3 = np.sum(c == True)/(4*144*40)
    
#     return rel, log, rms, threshold1, threshold2, threshold3


def accuracy(logits, gt_label, scope='accuracy'):
  with tf.variable_scope(scope):
    dim = logits.get_shape()[3].value
    logits = tf.reshape(logits, [-1, dim])
    gt_label = tf.reshape(gt_label, [-1])
    #correct_prediction = tf.equal(tf.argmax(logits, 1), gt_label)
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "int64"))
    pred_label = tf.argmax(logits, 1)
    acc = 1.0 - tf.nn.zero_fraction(tf.cast(tf.equal(pred_label, gt_label), tf.int32))
  return acc
    
    


