import sys
sys.path.append("./src")

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from Dataloader import dataGenerator, multiDataGenerator
import argparse
import random
import tensorflow as tf
import glob
import os
import time


# OpenDataset-10dB-SC-20211025_161239-best.h5
# OpenDataset-10dB-SC_FTA-20211025_163218-best.h5
# Args define
parser = argparse.ArgumentParser()
parser.add_argument("-path",
                    help="Project path.", default='./')
parser.add_argument("-model",
                    help="Model path.", default='saved_models/OpenDataset-10dB-SC_FTA-20211025_163218-best.h5')
parser.add_argument("-gpu",
                    help="GPU id to use.", default='1')
args = parser.parse_args()

# Params Settings
useGPU = args.gpu
model_path = args.model

# Runing enviroment settings
assert K.image_data_format() == 'channels_last'
# Multi GPUs Settings
os.environ['CUDA_VISIBLE_DEVICES'] = useGPU
if useGPU:
    gpu_ids = list(map(int, useGPU.split(',')))
# TF Settings & Determinism
seeds = 100
random.seed(seeds)
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(seeds)
tf.compat.v1.set_random_seed(seeds)
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(
    graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


# Evaluate Model Performance
model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer="Adam",
              metrics=['accuracy'])
model.summary()
input_shape = list(model.layers[0].input_shape)


# Inference Time
times = int(1e5)
input_shape[0] = 1
input_sample = K.variable(np.random.rand(*input_shape))

start_time = time.time()
for _ in range(times):
    _ = model(input_sample)
end_time = time.time()
print('\x1b[6;30;42m' + "Avg infer time:" +
      str((end_time - start_time)/times*1000.0) + "ms" + '\x1b[0m')


# Training Time
times = int(1e3)
batch_size = 512
input_shape[0] = batch_size

input_sample = K.variable(np.random.rand(*input_shape))
input_label = K.variable(np.random.rand(batch_size, 11))

start_time = time.time()
for _ in range(times):
    _ = model.train_on_batch(input_sample, input_label)
end_time = time.time()
print('\x1b[6;30;42m' + "Avg train time:" + str((end_time -
      start_time)/batch_size/times*1000.0) + "ms" + '\x1b[0m')
