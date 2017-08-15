#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName:slides_09
# Created by zhuchao at 2017/8/14 上午10:16
# purpose: Slides_09 TensorFlow Input Pipeline


import tensorflow as tf
import threading

## Agenda
## Data Readers Revisited
## TFRecord
## Variable initializer
## Graph Collection
## Style Transfer

## Queues
# tf.Session objects are designed to multithreaded. -> can run ops in parallel

## tf.Coordinator and tf.train.QueueRunner

## QueueRunner
# create a number of threads cooperating to enqueue tensors in the same queue

## Coordinator
# help multiple threads stop together and report exceptions to a program that waits for them to stop

## different queues in tf
## tf.FIFOQueue
## tf.RandomShuffleQueue
## tf.PaddingFIFOQueue
## tf.PriorityQueue

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0., 0., 0.],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    q_inc.run()
    q_inc.run()
    q_inc.run()
    q_inc.run()

## Create a queue
## tf.FIFOQueue(capacity,min_after_dequeue,dtypes,shapes=None,names=None)

## tf.Coordinator
## Can be used to manage the threads you created without queues

## Three ways to read in data
## Through tf.constant(make everything a constant) -> NO
## Feed dict Slow when client and workers are on different machines -> Maybe
## Data readers Readers allow us to load data directly into the worker process

## Different Readers for different file types
## tf.TextLineReader Outputs the lines of a file delimited by newlines
## tf.FixedLengthRecordReader
## tf.WholeFileReader
## tf.TFRecordReader
## tf.ReaderBase


## Read in files from queues
# filename_queue = tf.train.string_input_producer(["file0.csv","file1.csv"])
# reader = tf.TextLineReader()
# key,value = reader.read(filename_queue)

filename_queue = tf.train.string_input_producer(["../data/heart.csv"])
reader = tf.TextLineReader(skip_header_lines=1)

key, value = reader.read(filename_queue)
print(key, value)
# with tf.Session() as sess:
#     ## tf.train.string_input_producer creates a FIFOQueue under the hood, so to run the queue, we’ll need tf.Coordinator and tf.QueueRunner.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for _ in range(10):
#         key, value = reader.read(filename_queue)
#         key1, value1 = sess.run([key, value])
#         print(key1, value1)
#     coord.request_stop()
#     coord.join(threads)
#
with tf.Session() as sess:
    ## tf.train.string_input_producer creates a FIFOQueue under the hood, so to run the queue, we’ll need tf.Coordinator and tf.QueueRunner.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for _ in range(1):  # generate 1 example
        key, value = sess.run([key, value])  ## 计算得到的 key value 都是byte
        print(type(key), type(value))
        print(value)
        print(key)
    coord.request_stop()
    coord.join(threads)



## TFReacord
## TensorFlow's binary file format a serialized tf.train.Example protobuf object

## Why binary?
## make better use of disk cache
## faster to move around
## can store data of different types(so you can put both images and labels in one place)

## Convert normal files to TFRecord
## Super easy

## Loss functions revisited
## Content loss
## To measure the content loss between the feature map in the content layer of the generated image and the content image
## Style loss
## To measure the style loss between the feature maps in the style layers of the generated image and the style image

## In practice you rarely use a queue by itself,but always with string_input_producer in more details in a little bit

## you can use tf.Coordinator to manage threads created by python

# def my_loop(coord):
#     x = 0
#     while not coord.should_stop():
#         print("still looping",x)
#         x = x + 1
#         if x == 1000:
#             coord.request_stop()
#
# coord = tf.train.Coordinator()
#
# threads = [threading.Thread(target=my_loop,args=(coord,)) for _ in range(10)]
#
# for t in threads:
#     t.start()
# coord.join(threads)
