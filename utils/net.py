'''
Network architecture
'''
import tensorflow as tf
import tflearn


def joint_seg_net(img_inp, NUM_POINTS, NUM_CLS):
    x=img_inp
    #128 128
    x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #64 64
    x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #32 32
    x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    #16 16
    x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,128,activation='relu',weight_decay=1e-3,regularizer='L2')
    x=tflearn.layers.core.fully_connected(x,NUM_POINTS*(3+NUM_CLS),activation='linear',weight_decay=1e-3,regularizer='L2')
    x=tf.reshape(x,(-1,NUM_POINTS,3+NUM_CLS))
    x_pts, x_cls = tf.split(x, [3,NUM_CLS], axis=2)
    return x_pts, x_cls
