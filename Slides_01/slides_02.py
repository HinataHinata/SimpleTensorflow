#!/usr/bin/env python3
# -*- coding:utf-8 -*-

' simple tensorflow demo '

__author__ = 'zhuchao'

# Visualize tensorflow program with TensorBoard

import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)
with tf.Session() as sess:
    # add this line to use TensorBoard
    writer = tf.summary.FileWriter("./graphs",sess.graph)
    result = sess.run(x)
print(result)
writer.close()

# 在 terminal 中运行 py 并且执行 $ tensorflow --logdir="./graphs" --port 6006 即可在 localhost:6006 中看到 Graph

# 可以在创建变量的时候，给变量命名，这样在图中就能看到输入和操作的名字。

import tensorflow as tf
a = tf.constant(2,name="a")
b = tf.constant(3,name="b")
x = tf.add(a,b,name="add")
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    result = sess.run(x)
    print(result)
    writer.close()

# 更加详细的 constant 创建方式 tf.constant(value,dtype=None,shape=None,name='Const',varify_shape=False)

import tensorflow as tf
a = tf.constant([2, 2],name="a")
b = tf.constant([0, 1],name="b")
x = tf.add(a,b,name="add")
y = tf.multiply(a,b,name="mul")
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs",sess.graph)
    x,y = sess.run([x,y])
    print(x,y)
    writer.close()

# 使用特定类型的数值初始化变量
# tf.zeros(shape,dtype=tf.float32,name=None)

a = tf.zeros([2,3],tf.int32)
print(a)
with tf.Session() as sess:
    result = sess.run(a)
    print(result)

# 根据已有的变量初始化变量
# tf.zeros_like(input_tensor,dtype=None,name=None,optimize=True)
# input_tensor is [[0,1],[2,3],[4,5]]
# tf.zeros_like(input_tensor) ==> [[0,0],[0,0],[0,0]]

# tf.ones(shape,dtype=tf.float32,name=None)
# tf.ones_like(input_tensor,dtype=None,name=None,optimize=True)

# 使用特定值初始化变量
# tf.fill(dims,value,name=None)
# tf.fill([2,3],8) ==> [[8,8,8],[8,8,8]]

# tf.linspace(start,stop,num,name=None) # 创建梯度自增数组
