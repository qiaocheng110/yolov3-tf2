#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/13/20 4:53 PM
# @Author  : joshon
# @Site    : hsae
# @File    : test.py
# @Software: PyCharm

import tensorflow as tf
# @tf.function
def f():
    a = tf.constant([[10,10],[11.,1.]])
    x = tf.constant([[1.,0.],[0.,1.]])
    b = tf.Variable(12.)
    y = tf.matmul(a, x) + b
    print("PRINT: ", y)
    tf.print("TF-PRINT: ", y)
    return y

f()

