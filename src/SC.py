import argparse
import os
import random
import time

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils import multi_gpu_model
from sklearn.metrics import classification_report

from Dataloader import DataGenerator
from Models import ScModel
from Utils import (Mkdir, PlotConfusionMatrix, PlotTrainCurve,
                   RedirectLog, WriteHistory, EnvSettings)


"""
    Args define
"""
parser = argparse.ArgumentParser()
parser.add_argument("-dataset",
                    help="Dataset to run.", default='RadioML2016.10a')
parser.add_argument("-path",
                    help="Project path.", default='./')
parser.add_argument("-snr", type=int,
                    help="SNR to run.", default=10)
parser.add_argument("-lr", type=float,
                    help="Learning rate to run.", default=5e-4)
parser.add_argument("-att",
                    help="Use attention.", default="FTA")
parser.add_argument("-train", action="store_true",
                    help="True for train, False for test.", default=True)
parser.add_argument("-gpu",
                    help="GPU id to use.", default='3')
args = parser.parse_args()


"""
    Params Settings
"""
use_gpu = args.gpu
dataset = args.dataset
snr = args.snr
use_att = args.att
def_lr = args.lr
is_train = args.train

# Training params
img_width, img_height = 100, 100
epochs = 200
batch_size = 20
seeds = 100

# Path
dataset_path = args.path + '/SpectDataset'
output_path = args.path + '/Outputs'

time_prefix = time.strftime("-%Y%m%d_%H%M%S")
attention_prefix = "_" + use_att if use_att else ""

model_name = os.path.basename(__file__).split('.py')[0] + attention_prefix
last_model_path = output_path + '/' + dataset + \
    '-' + str(snr) + 'dB-' + model_name + time_prefix + '-last.h5'
best_model_path = output_path + '/' + dataset + \
    '-' + str(snr) + 'dB-' + model_name + time_prefix + '-best.h5'
history_path = output_path + '/' + dataset + \
    '-' + str(snr) + 'dB-' + model_name + time_prefix + '-history.log'
log_path = output_path + '/' + dataset + \
    '-' + str(snr) + 'dB-' + model_name + time_prefix + '.log'

train_data_dir = dataset_path + '/' + dataset + '/train/' + str(snr)
validation_data_dir = dataset_path + '/' + dataset + '/val/' + str(snr)
test_data_dir = dataset_path + '/' + dataset + '/test/' + str(snr)

Mkdir(output_path)
RedirectLog(log_path)

print("Using GPU:", use_gpu)
print("Dataset:", dataset)
print("SNR:", snr)
print("Use Attention", use_att)
print("Learning rate:", def_lr)
print("Is train? ", is_train)
print("Runing model:", model_name)

print('train_data_dir:', train_data_dir)
print('validation_data_dir:', validation_data_dir)
print('test_data_dir:', test_data_dir)

# Env settings
gpu_ids = EnvSettings(use_gpu, seeds=seeds)


'''
    Dataloader
'''
train_generator, train_nums = DataGenerator(train_data_dir,
                                            target_size=(
                                                img_width, img_height),
                                            batch_size=batch_size,
                                            shuffle=True, seed=seeds)

val_generator, val_nums = DataGenerator(validation_data_dir,
                                        target_size=(img_width, img_height),
                                        batch_size=batch_size,
                                        shuffle=True, seed=seeds)

test_generator, test_nums = DataGenerator(test_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          shuffle=False)

target_names = list(train_generator.class_indices.keys())
print('Target Names:', len(target_names), target_names)

input_shape = (img_width, img_height, 3)
ch_axis = -1
print('Input hape:', input_shape)


'''
    Build Model
'''
# Build classification model
model = ScModel(input_shape=input_shape, attention=use_att,
                categories_num=len(target_names))
model.summary()

# Optimizer
# optm = keras.optimizers.Adam(lr=def_lr)
optm = keras.optimizers.RMSprop(lr=def_lr, rho=0.9, epsilon=1e-6)
# optm = keras.optimizers.Adadelta()
# optm = keras.optimizers.SGD()

# Multi GPU Model Settings
if use_gpu and len(gpu_ids) > 1:
    model = multi_gpu_model(model, gpus=len(gpu_ids))

# Compile Model
model.compile(loss='categorical_crossentropy',
              optimizer=optm,
              metrics=['accuracy'])


'''
    Train Model
'''
if is_train:
    history = model.fit_generator(
        train_generator,
        shuffle=False,
        steps_per_epoch=train_nums // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                best_model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=15, verbose=1, mode='auto'),
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, verbose=1, mode='auto')
        ],
        validation_steps=val_nums // batch_size,
        verbose=2
    )
    model.save(last_model_path)
    # PlotTrainCurve(history)
    WriteHistory(history, path=history_path)


'''
    Evaluate Model Performance
'''
# Evaluate LAST performance
model.load_weights(last_model_path)
score_last = model.evaluate_generator(
    test_generator, steps=test_nums // batch_size)

# Evaluate BEST performance
model.load_weights(best_model_path)
score_best = model.evaluate_generator(
    test_generator, steps=test_nums // batch_size)

# Choose weights with better accuracy
if score_best[1] > score_last[1]:
    model.load_weights(best_model_path)
    print('SNR:', snr, 'Choose Weights: Best, Acc:', round(score_best[1], 4))
else:
    model.load_weights(last_model_path)
    print('SNR:', snr, 'Choose Weights: Last, Acc:', round(score_last[1], 3))


'''
    Test Model
'''
if not is_train:
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict_generator(test_generator), axis=1)
    assert len(y_true) == len(y_pred)

    # PlotConfusionMatrix(target_names, y_true, y_pred, save=True)

    print(classification_report(y_true, y_pred, target_names=target_names))
