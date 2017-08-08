#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName:slides_05
# Created by zhuchao at 2017/8/8 下午4:02
# purpose: implemtent slides-05

import tensorflow as tf

x = tf.Variable(2.0)
y = 2.0 * (x ** 3)  # 表示 y = 2.0 * x^3
z = 3.0 + y ** 2  # 表示 z = 3.0 + y^2

grad_z = tf.gradients(z, [x, y])  # 求 z 相对与x与y的导数。
with tf.Session() as sess:
    sess.run(x.initializer)
    print(sess.run(y))
    print(sess.run(z))
    print(sess.run(grad_z))  # 【768 32】

## saves graph's variables in binary files
# tf.trian.Saver

## Save sessions,not graphs
# tf.train.Saver.save(sess,save_path,global_step=None...)

## Save parameters after 1000 steps

# define model

# create a saver object
saver = tf.train.Saver()

# lauch a session to compute the grahp

# with tf.Session() as sess:
#     # actual training loop
#     for step in range(training_steps):
#         sess.run([optimizer])
#
#         if (step +1) % 1000 == 0:
#             saver.save(sess,'checkpoint_directory/model_name',global_step=model.global_step)

## Global step is Very common in TensorFlow program
# self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')

## need to tell optimizer to increment global step
# self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,global_step = self.global_step)

## tf.train.Saver only save variables not graph. Checkpoints map variable names to tensors

## restore variables
# saver.restore(sess,'checkpoints/name_of_the_checkpoint')

## Restore the latest checkpoint
## 1 checkpoint keeps track of the latest checkpoint 2 Safeguard to restore checkpoints only when there are chekcpoints
# ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
# if ckpt and ckpt.model_checkpoint_path
#   saver.restore(sess,ckpt.model_checkpoint_path)

## Control Randomization
## Op level random seed
# my_var = tf.Variable(tf.truncated_normal((-1.0,1.0),stddev=0.1,seed=0))

my_var = tf.Variable(tf.truncated_normal((-1.0, 1.0), stddev=0.1, seed=0))

# with tf.Session() as sess:
#     for i in range(100):
#         sess.run(my_var)

# c = tf.random_uniform([],-10,10,seed=2)
# with tf.Session() as sess:
#     for i in range(100):
#         print(sess.run(c))
