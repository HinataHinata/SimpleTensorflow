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

# 在 terminal 中运行 py 并且执行 $ tensorboard --logdir="./graphs" --port 6006 即可在 localhost:6006 中看到 Graph

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

# 产生随机数
# tf.random_normal()
# tf.truncated_normal()
# tf.random_uniform()
# tf.random_shuffle()
# tf.random_crop()
# tf.multinomial()
# tf.random_gamma()

# tf.set_random_seed(seed)

# operations 一些数值的运算
# a = tf.constant([3,6])
# b = tf.constant([2,2])
# tf.add(a,b) # >> [5,8]
# tf.add_n(a,b,b) # >> [7,10] a+b+b
# tf.mul(a,b) # >> [6,12] [ a[1]*b[1], a[2]*b[2] ]
# tf.matmul(tf.reshape(a,[1,2]),tf.reshape(b,[2,1])) # >> 18 mat mul
# tf.div(a,b) # >> [1,3]
# tf.mod(a,b) # >> [1,0]

# tensorflow 和 numpy 数据类型一致
import numpy as np
import tensorflow as tf
print(tf.int32 == np.int32) # True
print(tf.float32 == np.float32) # True

# Do not use Python native types for tensors because TensorFlow has to infer Python type

# see constant in the graph's definition
c1 = tf.constant([1.0, 2.0],name='c1')
with tf.Session() as sess:
    print(sess.graph.as_graph_def())

# this makes loading graphs expensive when constants are big

# only use constants for primitive tpyes.Use variables or readers for more data that requires more memory

# Variables
# create variable a with scalar value
a = tf.Variable(2,name='scalar')
b = tf.Variable([2, 3],name='vector')
#create variable c as 2*2 matrix
c = tf.Variable([[0,1],[2,3]],name='matrix')
W = tf.Variable(tf.zeros([784,10]))

# tf.constant is an op tf.Variable is a class

# You have to initialize your variables. The easiest way is initializing all variables at once
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# Initialize only a subset of variable
init_ab = tf.variables_initializer([a,b],name='init_ab')
with tf.Session() as sess:
    sess.run(init_ab)

# Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
    sess.run(W.initializer)

## Eval() a variable
W = tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W) # <tf.Variable 'Variable_1:0' shape=(700, 10) dtype=float32_ref>

W = tf.Variable(tf.truncated_normal([700,10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())

## tf.Variable.assign()

W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval()) # 10

# W.assign(100) doesn't assign the value to W.It creates an assign op,and that op needs to be run to take effect.
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(assign_op)
    print(W.eval()) # 100

# You don't ned to initialize variable because assign_op does it for you
# In fact,initializer op is the assign op that assigns the variable's initial value to the variable itself
W = tf.Variable(10)
assign_op1 = W.assign(100)
with tf.Session() as sess1:
    sess1.run(assign_op1)
    print(W.eval())

my_val = tf.Variable(2,name='my_val')
my_val_times_two = my_val.assign(2 * my_val)
with tf.Session() as sess:
    sess.run(my_val.initializer)
    sess.run(my_val_times_two) # 4
    sess.run(my_val_times_two) # 8
    sess.run(my_val_times_two) # 16  it assign 2* my_val to a every time my_val_times_two is effected

## assign_add and assign_sub
my_val = tf.Variable(10)
with tf.Session() as sess:
    sess.run(my_val.initializer)
    sess.run(my_val.assign_add(10))
    print(my_val.eval()) # 20
    sess.run(my_val.assign_sub(2))
    print(my_val.eval()) # 18

## Each session maintains its own copy of variable
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)
print('--------')
print(sess1.run(W.assign_add(10))) # 20
print(sess2.run(W.assign_sub(2))) # 8

sess1.close()
sess2.close()

## Use a variable to initialize another variable
# W = tf.Variable(tf.truncated_normal([700,10]))
# U = tf.Variable(2 * W) # not so safe

W = tf.Variable(tf.truncated_normal([700,10]))
U = tf.Variable(2 * W.initialized_value()) # ensure that W is initialized before its value is used to initialize U

## Session vs InteractiveSession
# The InteractiveSession makes itself the default
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b
print(c.eval())
sess.close()

## Control Dependencies   EXP: tf.Graph.control_dependencies(control_inputs)
# defines which ops should be run first
# your graph g has 5 ops:a,b,c,d,e
# with g.control_dependencies([a,b,c]): # d and e will only run after a b c have executed.

## Project Speed Dating

# A quick reminder
# A TF program often has 2 phases(阶段)
# 1 Assemble a graph
# 2 Use a session to execute operations in the graph

# => Can assemble the graph first without knowing the value needed for computation

# Placeholders
# tf.placeholder(dtype,shape=None,nam=None)
# create a placeholder of type float 32-bit,shape is a vector of 3 elements
a = tf.placeholder(tf.float32,shape=[3])
# create a constant of type float 32-bit shape is a vector of 3 element
b = tf.constant([5,5,5],tf.float32)
# use the placeholder as you would a constant or a variable
c = a+b # short for tf.add(a,b)

with tf.Session() as sess:
    # feed [1,2,3] to placeholder a via the dict(a:[1,2,3])
    # fetch value of c
    print(sess.run(c,{a:[1,2,3]})) # the tensor a is the key,not the string 'a'
    # >>> [6,7,8]

# shape = None means that tensor of any shape will be acceptd as value for placeholder
# shape = None is easy to construct graphs,but nightmarish for debugging
# shape = None also breaks all following shape inference,which makes many ops not work because they expect certain rank.

# What if want to feed multiple data points in?
# We feed all the values in one at a time
# with tf.Session() as sess:
#     for a_value in list_of_values_for_a:
#         print(sess.run(c,{a:a_value}))
#

# tf.Graph.is_feedable(tensor) # True if and only if tensor is feedable

# Feeding values to TF ops

# create operations,tensor,etc
a = tf.add(2,5)
b = tf.multiply(a,3)
with tf.Session() as sess:
    replace_dict = {a:15}
    print(sess.run(b,feed_dict=replace_dict)) # 45
# Extremely helpful for teting too

## lazy loading -- Defer creating/initializing an object until it is needed

# Normal loading
x1 = tf.Variable(10,name='x1')
y1 = tf.Variable(20,name='y1')
z1 = tf.add(x1,y1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/12',sess.graph)
    for _ in range(10):
        sess.run(z1)
    writer.close()

## lazy loading
x = tf.Variable(10)
y = tf.Variable(20)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/12',sess.graph)
    for _ in range(10):
        sess.run(tf.add(x,y)) # someone decides to be clever to save one line of code
    writer.close()

print(tf.get_default_graph().as_graph_def()) # 打印变量












