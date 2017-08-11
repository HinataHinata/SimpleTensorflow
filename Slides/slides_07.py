#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName:slides_07
# Created by zhuchao at 2017/8/9 下午6:38
# purpose: Convnets in TensorFlow


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils

batch_size = 128
n_epochs = 2
learning_rate = 0.01
## Understanding convolutions
## Convolutions in maths and physics
## a function derived from two given functions by integration that expresses how the shape of one is modified by the other
## Conversion in neural network
## a function derived from two given functions by element-wise multiplication that express how the value and shape of one is modified by the other

## we can use one single convolutional layer to modify a certain image


# tf.nn.conv2d(input=,filter=,strides=,padding=,use_cudnn_on_gpu=,data_format=,name=)

## Convolutions in neural networks
## In training,we don't specify kernel. We learn kernels.

mnist = input_data.read_data_sets("../data/mnist",one_hot=True)
print("init mnist...")

num_examples = mnist.train.num_examples
print(num_examples)
n_batches = int(mnist.train.num_examples / batch_size)
print(n_batches,batch_size)
iamgebits = 0
for i in range(n_batches):
    X_batch,Y_batch = mnist.test.next_batch(batch_size)
    print(X_batch,"\n============ Y_batch ==========\n",Y_batch)
    imagebits = X_batch[0]
    print(len(X_batch),len(Y_batch),len(X_batch[0]),len(Y_batch[0])) # 纬度 128 128 784 10
    print("===========")

## Variable scope
## similar to a namespace
## in variable scope,we don't create variable using tf.Variable,but instead use tf.get_variable()
# tf.get_variable(<name>,<shape>,<initializer>)
# If a variable with that name already exists in that variable scope, we use that variable.
# If a variable with that name doesn’t already exists in that variable scope, TensorFlow creates a new variable.
# This setup makes it really easy to share variables across architecture.

## Declare a variable scope
## with tf.variable_scope("conv1") as scope:

image = []
with tf.variable_scope('conv1') as scope:
    w = tf.get_variable('weight',[5,5,1,32])
    b = tf.get_variable('biases',[32],initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(image,w,strides=[1,1,1,1],padding='SAME')
    conv1 = tf.nn.relu(conv + b,name=scope.name)

with tf.variable_scope('conv2') as scope:
    w = tf.get_variable('weights',[5,5,32,64])
    b = tf.get_variable('biases',[64],tf.random_normal_initializer())
    conv = tf.nn.conv2d(conv1,w,strides=[1,1,1,1],padding='SAME')
    conv2 = tf.nn.relu(conv + b,name=scope.name)




