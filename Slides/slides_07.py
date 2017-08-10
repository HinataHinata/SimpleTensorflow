#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName:slides_07
# Created by zhuchao at 2017/8/9 下午6:38
# purpose: Convnets in TensorFlow

## Understanding convolutions
## Convolutions in maths and physics
## a function derived from two given functions by integration that expresses how the shape of one is modified by the other
## Conversion in neural network
## a function derived from two given functions by element-wise multiplication that express how the value and shape of one is modified by the other

## we can use one single convolutional layer to modify a certain image

import tensorflow as tf

# tf.nn.conv2d(input=,filter=,strides=,padding=,use_cudnn_on_gpu=,data_format=,name=)
