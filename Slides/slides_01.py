#!/usr/bin/env python3
# -*- coding:utf-8 -*-

' simple tensorflow demo '

__author__ = 'zhuchao'

import tensorflow as tf

# first tensorflow program

hello = tf.constant('Hello,Tensorflow')
sess = tf.Session()
result = sess.run(hello)
print(result) # result b'Hello,Tensorflow'
print(result.decode('utf-8')) # Hello,Tensorflow
print(result.decode('gb2312')) # Hello,Tensorflow
sess.close() # 关闭 Session

x = 3
y = 5
op1 = tf.add(x,y)
op2 = tf.multiply(x,y)
op3 = tf.add(op1,op2)
with tf.Session() as sess:
    result = sess.run(op3)
print(result) # (3+5)+(3*5) = 23


# useless Graph 中 不需要计算的部分在 session.run() 的时候不真正的执行 已节约计算资源。
x = 2
y = 3
add_op = tf.add(x,y)
useless = tf.multiply(x,add_op)
mul_op = tf.multiply(x,y)
pow_op = tf.pow(add_op,mul_op)
with tf.Session() as sess:
    result = sess.run(pow_op)
print(result) # 15625

#session.run()可以接收list参数 结果以list返回
x = 2
y = 3
add_op = tf.add(x,y)
useless = tf.add(x,add_op)
mul_op = tf.multiply(x,y)
pow_op = tf.pow(add_op,mul_op)
with tf.Session() as sess:
    result_pow,result_useless = sess.run([pow_op,useless])
print(result_pow,result_useless) # 15627 7

# tensorflow 有可能会将图拆分成几块，分配到不同的CPU，GPU，或者是不同的设备上去运行。
# 虽然有办法创建多个 Graph 但是在tensorflow 中尽量只创建一个 Graph
# 01 多个图的计算需要多个 session 每个 session 默认都会使用所有的可用计算资源。 02 多个 Graph 之间不太容易传递数据 03更好的选择是在一个图中使用不连接的子图

# tf.Graph() 支持创建多个图 在使用多个图时，需要先将要使用的图设置为default。
g = tf.Graph()
with g.as_default():
    x = tf.add(3,5)
# sess = tf.Session(graph=g)
# result = sess.run(x)
# sess.close()
with tf.Session(graph=g) as sess:
    result = sess.run(x)
print(result)

# 获取默认的 Graph g1 = tf.get_default_graph()
