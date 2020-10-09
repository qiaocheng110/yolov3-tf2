#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/13/20 4:53 PM
# @Author  : joshon
# @Site    : hsae
# @File    : test.py
# @Software: PyCharm
from tensorflow.keras.layers import Lambda ,Input
import tensorflow as tf
import numpy as np
import keras.backend as K
import cv2

# @tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y


tt1 = K.variable(np.array([[[0, 22], [29, 38]], [[49, 33], [5, 3]], [[8, 8], [9, 9]]]))
tt2 = K.variable(np.array([[[55, 47], [88, 48]], [[28, 10], [15, 51]], [[5, 5], [6, 6]]]))

t1 = K.variable(np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]]))
t2 = K.variable(np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]]))

dd3 = K.concatenate([tt1 , tt2] , axis=0)

print("111")
# def slice(x,index):
#     return x[:,:,index]
#
# a = Input(shape=(4,2))
# x1=Lambda(slice,output_shape=(4,1),arguments={'index':0})(a)
# x2 = Lambda(slice,output_shape=(4,1),arguments={'index':1})(a)
# Lambda(slice,output_shape=(4,1))([a,0])
#
# x1 = tf.reshape(x1,(4,1,1))
# x2 = tf.reshape(x2,(4,1,1))