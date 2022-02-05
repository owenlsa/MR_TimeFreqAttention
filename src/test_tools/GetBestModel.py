import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import glob
import argparse
from utils import (Mkdir, PlotConfusionMatrix, PlotTrainCurve,
                   RedirectLog, WriteHistory)
from sklearn.metrics import classification_report
from keras import backend as K
from dataloader import DataGenerator, MultiDataGenerator
from keras.models import load_model
import sys
sys.path.append("..")


"""
    Args define
"""
parser = argparse.ArgumentParser()
parser.add_argument("-dataset",
                    help="Dataset to run.", default='OpenDataset_Filter_yuan')
parser.add_argument("-method",
                    help="Method.", default='SC')
parser.add_argument("-filt_dataset",
                    help="Second dataset to run, for dual-channel input.", default='')
parser.add_argument("-path",
                    help="Project path.", default='/home/gongjuren/disk2/lsa/RobustMR')
parser.add_argument("-models",
                    help="Models path.", default='/home/gongjuren/disk2/lsa/RobustMR/saved_models')
parser.add_argument("-snr", type=int,
                    help="SNR to run.", default=-4)
parser.add_argument("-gpu",
                    help="GPU id to use.", default='2')
args = parser.parse_args()


"""
    Params Settings
"""
use_gpu = args.gpu
dataset = args.dataset
dataset_filtered = args.filt_dataset
snr = args.snr
modelsList = glob.glob(args.models + "/" + dataset +
                       "-" + str(snr) + "dB-" + args.method + "-*.h5")

# Training params
img_width, img_height = 100, 100

# Path
dataset_path = args.path + '/SpectDataset'

test_data_dir = dataset_path + '/' + dataset + '/test/' + str(snr)
filtered_test_data_dir = dataset_path + '/' + \
    dataset_filtered + '/test/' + str(snr)


'''
    Runing enviroment settings
'''
assert K.image_data_format() == 'channels_last'
# Multi GPUs Settings
os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu
if use_gpu:
    gpu_ids = list(map(int, use_gpu.split(',')))
# TF Settings & Determinism
seeds = 100
# tfdeterminism.patch()
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


'''
    Dataloader
'''
# one channel DataGenerator
test_generator, test_nums = DataGenerator(test_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=1,
                                          shuffle=False)


def TestGene():
    # dual channel DataGenerator
    return MultiDataGenerator(test_data_dir, filtered_test_data_dir, target_size=(img_width, img_height), batch_size=1, shuffle=False)


target_names = list(test_generator.class_indices.keys())


'''
    Evaluate Model Performance
'''
bestModelPath = ''
best_acc = 0.0
for oneModelPath in modelsList:
    model = load_model(oneModelPath)
    if len(model.input_shape) == 2:
        score_last = model.evaluate_generator(TestGene(), steps=test_nums)
    else:
        score_last = model.evaluate_generator(test_generator, steps=test_nums)
    print("Tested Model:", oneModelPath.split('/')[-1], "Acc:", score_last[1])
    if score_last[1] > best_acc:
        bestModelPath = oneModelPath
        best_acc = score_last[1]

print('SNR:', snr)
print("Choose Model:", bestModelPath)
print('Acc:', best_acc)


'''
    Test Model
'''
model = load_model(bestModelPath)
y_true = test_generator.classes
if len(model.input_shape) == 2:
    y_pred = np.argmax(model.predict_generator(
        TestGene(), steps=test_nums, verbose=1), axis=1)
else:
    y_pred = np.argmax(model.predict_generator(
        test_generator, steps=test_nums, verbose=1), axis=1)
assert len(y_true) == len(y_pred)

#PlotConfusionMatrix(target_names, y_true, y_pred, save="confMat.svg")

print(classification_report(y_true, y_pred, target_names=target_names, digits=3))
