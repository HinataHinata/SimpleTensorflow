#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName:slides_06
# Created by zhuchao at 2017/8/9 上午10:36
# purpose: tensorflow tutorial slides_06 Convolutional Neural Networks + Neural Style Transfer

## What is a Convolutional Neural Net
## https://zh.wikipedia.org/wiki/卷积神经网络
# CNN 是一种前馈神经网络，他的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。
# 卷积神经网络由一个或多个卷积层和顶端的全连通层（对应经典的神经网络）组成，同时也包括关联权重和池化层（pooling layer）。
# 这一结构使得卷积神经网络能够利用输入数据的二维结构。与其他深度学习结构相比，卷积神经网络在图像和语音识别方面能够给出更优的结果。
# 这一模型也可以使用反向传播算法进行训练。
# 相比较其他深度、前馈神经网络，卷积神经网络需要估计的参数更少，使之成为一种颇具吸引力的深度学习结构。

## Convolution Layer  Convolve the filter with the image i.e. "slide over the image spatially,computing dot products"

##  Pooling Layer
## - makes the representations smaller and more manageable
## - operates over each activation map independently
## Max Pooling

## Case Study: LeNet-5
## Case Study: AlexNet
## Case Study: VGGNet
## Case Study: GoogLeNet
## Case Study: ResNet

## Question How does the chosen neuron respond to the image?
## 1 Feed image into net
## 2 Set gradient of chosen layer to all zero,except 1 for the chosen neuron
## 3 Backprop to image

## Visualizing CNN features:Gradient Ascent
## (Guided) backprop: Find the part of an image that a neuron responds to
## Gradient ascent: Generate a synthetic image that maximally activates a neuron

## Visualizing CNN features: Gradient Ascent
## 1 Initialize image to zeros
## Repeat:
## 2 Forward image to compute current scores
## 3 Set gradient of scores to be 1 for target class,0 for others
## 4 Backprop to get gradient on image
## 5 Make a small update to the image

## Feature Inversion
## Given a feature vector for an image, find a new image such that
## - Its features are similar to the given features
## -It "looks natural"(image prior regularization)

## Higher layers are less sensitive to changes in color,texture,and shape


## (Neural)Texture Synthesis
## Torch implementation
## https://github.com/jcjohnson/texture-synthesis

## Neural Texture Synthesis
## 1 Pretrain a CNN on ImageNet(VGG-19)
## 2 Run input texture forward through CNN,record activations on every layer;layer i gives feature map of shape Ci*Hi*Wi
## 3 At each layer compute the Gram matrix giving outer product of features
## 4 Initialize generated image from random noise
## 5 Pass generated image through CNN,compute Gram matrix on each layer
## 6 Compute loss:weighted sum of L2 distance between Gram matrices
## 7 Backprop to get gradient on image
## 8 Make gradient step on image
## 9 GOTO 5


## Neural Style Transfer
## Given a content image and a style image,find a new image that
## - Matches the CNN features of the content image(feature reconstruction)
## - Matches the Gram matrices of the style image(texture synthesis)
## Combine feature reconstruction from Mahendran et al with Neural Texture Synthesis from Gatys et al,using the same CNN

## 1 Pretrain CNN
## 2 Compute feature for content image
## 3 Compute Gram matrices from style image
## 4 Randomly initialize new image
## 5 Forward new image through CNN
## 6 Compute style loss(L2 distance between Gram matrices) and content loss(L2 distance between features)
## 7 Loss is weighted sum of style and content losses
## 8 Backprop to image
## 9 Take a gradient step
## GOTO 5

## implementation on Github:
## https://github.com/jcjohnson/neural-style

## Resizing style image before running style transfer algorithm can transfer different types of features
## Perform style transfer only on the luminance channel Copy colors from content image

## Style Transfer on Video






