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


# with tf.Session() as sess:
#     for i in range(100):
#         sess.run(my_var)

# c = tf.random_uniform([],-10,10,seed=2)
# with tf.Session() as sess:
#     for i in range(100):
#         print(sess.run(c))

## Op level seed: each op keeps its own seed
c = tf.random_uniform([], -10, 10, seed=2)
d = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c))  # 3.57493
    print(sess.run(d))  # 3.57493

## Graph level seed
# seed = 2
# tf.set_random_seed(seed)


## problem with feed_dict
# [Storage] -> [Client] -> [Worker]
# Slow when Client and workers are on different machines

## Data Readers
# [Storage] -> [Worker]
# Readers allow us to load data directly into the worker process

## Data Readers Ops that return different values every time you call them (Think Python's generator)
## Different Readers for different file types
# tf.TextLineReader Outputs the lines of a file delimited by newlines E.g. text files CSV files
# tf.FixedLengthRecordReader Outputs the entire file when all files have same fixed lengths E.g. each MNIST file has 28*28 pixels,CIFAR-10 32*32*3
# tf.WholeFileReader Outputs the entire file content
# tf.TFRecordReader Reads samples from TensorFlow's own binary format (TFRecord)
# tf.ReaderBase To allow you to create your own readers

## Read in files from queues
# filename_queue = tf.train.string_input_producer(["file0.csv","file1.csv"])
# reader = tf.TextLineReader()
# key,value = reader.read(filename_queue)

## tf.FIFOQueue
q = tf.FIFOQueue(3,"float")
init = q.enqueue_many(([0.,0.,0.],))
x = q.dequeue()
y = x+1
q_inc = q.enqueue([y])
sess1 = tf.Session()
print(init.run(session = sess1))
with tf.Session() as sess:
    print(init.run()) # None
    # print(init.run())
    # print(q_inc.run())
    # print(q_inc.run())
    # print(q_inc.run())
    # print(q_inc.run())

## Threads & Queues You can use tf.Coordinator and tf.QueueRunner to manage your queues
with tf.Session() as sess:
    # start populating the file queue.
    coord= tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)




