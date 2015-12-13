#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
import sys
from caffe.proto import caffe_pb2

# プロット設定
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['image.cmap'] = 'gray'

# 描画関数
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]),
               (0, padsize),
               (0, padsize)) + ((0, 0),
                                ) * (data.ndim - 3)
    data = np.pad(
        data, padding, mode='constant', constant_values=(padval, padval))

    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.show()


caffe.set_mode_cpu()
"""
model_root = '../.package/caffe_model/imagenet/'
proto_model = model_root + 'imagenet_feature.prototxt'
trained_model = model_root + 'caffe_reference_imagenet_model'
mean_model = model_root + 'imagenet_mean.binaryproto'
"""
model_root = '/home/dl-box/study/scene_emotion_degree/fine_tuning/obj_model/'
proto_model = model_root + 'feature.prototxt'
trained_model = model_root + 'finetuned.caffemodel'
mean_model = model_root + 'mean.binaryproto'

net = caffe.Net(proto_model,
                trained_model,
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))

mean_blob = caffe_pb2.BlobProto()
with open(mean_model) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(
mean_blob.data,
dtype=np.float32).reshape(
    (mean_blob.channels,
    mean_blob.height,
    mean_blob.width))

transformer.set_mean('data', mean_array)
transformer.set_raw_scale('data', 255)

image = caffe.io.load_image(sys.argv[1])
print image.shape
image = caffe.io.resize_image(image,[256,256])
print image.shape

#net.blobs['data'].reshape(1, 3, 32, 32)
net.blobs['data'].data[...] = transformer.preprocess(
    'data', image)
out = net.forward()

print([(k, v.data.shape) for k, v in net.blobs.items()])

# conv1の出力
features = net.blobs['conv1'].data[0, :]
vis_square(features, padval=1)

# conv2の出力
features = net.blobs['conv2'].data[0, :]
vis_square(features, padval=1)

# conv3の出力
features = net.blobs['conv3'].data[0, :]
vis_square(features, padval=1)

# conv4の出力
features = net.blobs['conv4'].data[0, :]
vis_square(features, padval=1)

# conv5の出力
features = net.blobs['conv5'].data[0, :]
vis_square(features, padval=1)

# 確率値の出力
print net.blobs['prob'].data


