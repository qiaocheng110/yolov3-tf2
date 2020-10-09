#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/24/20 3:09 PM
# @Author  : joshon
# @Site    : hsae
# @File    : tfRecordTest.py
# @Software: PyCharm
from builtins import tuple

import tensorflow as tf

import numpy as np
import IPython.display as disp


# dataset = tf.data.Dataset.range(5000)
# dataset = dataset.batch(8, drop_remainder=True)
# aa=list(dataset.as_numpy_iterator())




# data=tf.lookup.TextFileInitializer("data.txt",tf.string,0,tf.int64,1,delimiter=" ")
# table=tf.lookup.StaticHashTable(data,-1)
# print(table)


# dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
# dataset = dataset.filter(lambda x: x < 3)
# list(dataset.as_numpy_iterator())

# `tf.math.equal(x, y)` is required for equality comparison
# def filter_fn(x):
#   return tf.math.equal(x, 1)
# dataset = dataset.filter(filter_fn)
# list(dataset.as_numpy_iterator())
# array=tf.TensorArray(tf.int32, 1, dynamic_size=True)
# array1=tf.TensorArray(tf.int32, 1, dynamic_size=True)
#
# array.write([1])
import tensorflow as tf

# origin_embeddings = tf.constant([[ 1.1 ,  2.2 ,  3.3 ],
#                                  [ 4.4 ,  5.5 ,  6.6 ],
#                                  [ 7.7 ,  8.8 ,  9.9 ],
#                                  [10.10, 11.11, 12.12]])
# indice_updated_embeddings= tf.constant([[1.0, 2.0, 3.0],
#                                         [4.0, 5.0, 6.0]])
#
# indices = tf.constant([1, 3], dtype=tf.int32)
#
# test=tf.tensor_scatter_nd_update(origin_embeddings,tf.expand_dims(indices,1),indice_updated_embeddings)
#
# print("this is  bad things for you ")

# def fun(x):
#     a=tf.constant([x,x+1])
#     return a
# def fun1(x):
#     a=[]
#     a.append(tf.constant[x,x+10,x+100])
#     a.append(tf.constant[x,x+10,x+10])
#     return a
# dataset =tf.data.Dataset.range(1, 6)
# dataset = dataset.map(lambda x:x+1)
# a=list(dataset.as_numpy_iterator())
# print(a)
# dataset = tf.data.Dataset.range(100)
#
#
# dataset=tf.data.Dataset.from_tensor_slices(([1, 2, 3], [1,2,3]))
# dataset1=tf.data.Dataset.from_tensors([[1, 2, 3], [1,2,3]])
#
# elements = [(1, "foo"), (2, "bar"), (3, "baz")]
# elements1= [(1, 1), (2, 2), (3, 3)]
# dataset = tf.data.Dataset.from_generator(
#     lambda: elements, (tf.int32, tf.string))
#
# def f(x):
#   a=[]
#   a.append(x+1)
#   a.append(x+2)
#   return a;
#
# def g(x):
#   a=[]
#   a.append(x)
#   a.append(1)
#   return tuple(a);
#
# result = dataset.map(lambda x_int, y_str: (f(x_int),g(y_str)))
# list(result.as_numpy_iterator())

# dataset = tf.data.Dataset.range(3)
# def g(x):
#   return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
#
# def h(x):
#   return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64)
#
# def e(x):
#   num=[]
#   num.append(23)
#   num.append(["Foo", "Bar"])
#   num.append(np.array([1.0, 2.0], dtype=np.float64))
#   return num
#
# def f(x):
#   num=[]
#   num.append(23)
#   num.append(["Foo", "Bar"])
#   num.append(np.array([1.0, 2.0], dtype=np.float64))
#   return tuple(num)
#
# aa=dataset.map(lambda x:e(x))
# bb=dataset.map(f)
# indices = tf.constant([[0], [2]])
# updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
#                         [7, 7, 7, 7], [8, 8, 8, 8]],
#                        [[5, 5, 5, 5], [6, 6, 6, 6],
#                         [7, 7, 7, 7], [8, 8, 8, 8]]])
# tensor = tf.ones([4, 4, 4], dtype=tf.int32)
# print(tf.tensor_scatter_nd_update(tensor, indices, updates).numpy())
# indices = tf.constant([[4], [3], [1], [7],[1]])
# updates = tf.constant([9, 10, 11, 12,13])
# tensor = tf.ones([8], dtype=tf.int32)
# print(tf.tensor_scatter_nd_update(tensor, indices, updates))

@tf.function
def iterate_tensor(tensor):
    tf.print(type(tensor))  # Tensor
    (x1, x2, x3), (x4, x5, x6) = tensor
    return tf.stack([x2, x4, x6])
const1 = tf.constant(range(6), shape=(2, 3)) # EagerTensor
o = iterate_tensor(const1)
print(o)
print("=====data====")












