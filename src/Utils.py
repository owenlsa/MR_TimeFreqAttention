import itertools
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K


def RedirectLog(log_path):
    ''' 
    Redirect stdout and stderr to log file.

    Args
    - log_path: path of the output log file.
    '''
    f = open(log_path, 'w')
    sys.stdout = f
    sys.stderr = f


def Mkdir(path):
    ''' 
    Make dir.

    Args
    - path: path to create.
    '''
    os.makedirs(path, exist_ok=True)


def EnvSettings(use_gpu, seeds=100):
    '''
    Multi GPUs settings, TF settings and determinism
    '''
    assert K.image_data_format() == 'channels_last'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Multi GPUs Settings
    os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu
    if use_gpu:
        gpu_ids = list(map(int, use_gpu.split(',')))
    # TF Settings & Determinism
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seeds)
    tf.set_random_seed(seeds)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    return gpu_ids


def _ConfusionMatrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues, figsize=(8, 8)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args
    - cm : Value of confusion matrix.
    - classes : Class names in list.
    Optional
    - normalize : If True, shows percent, if False, shows exact numbers. Default: `True`
    - title: Title of figure. Default: `Confusion Matrix`
    - figsize: Size of the plt figure. Default: `(8, 8)`
    """
    plt.figure(figsize=figsize)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def PlotConfusionMatrix(target_names, y_true, y_pred, title='Confusion Matrix', normalize=True, figsize=(4.6, 4.6), save=""):
    '''
    Plot confusion matrix.

    Args
    - target_names: Class names in list.
    - y_true: True result.
    - y_pred: Predicted result.
    Optional
    - title: Title of figure. Default: `Confusion Matrix`
    - normalize: True: If True, shows percent, if False, shows exact numbers. Default: `True`
    '''
    plt_classes = target_names

    conf = np.zeros([len(plt_classes), len(plt_classes)])
    confnorm = np.zeros([len(plt_classes), len(plt_classes)])
    for i in range(0, len(y_true)):
        j = y_true[i]
        k = y_pred[i]
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(plt_classes)):
        confnorm[i, :] = conf[i, :]
        # / np.sum(conf[i,:])
    # _ConfusionMatrix(confnorm, labels=plt_classes)
    _ConfusionMatrix(confnorm.astype(int), classes=plt_classes,
                     normalize=normalize, title=title, figsize=figsize)
    if save:
        plt.savefig(save, format='svg')


def PlotTrainCurve(history, title='Training Performance'):
    ''' 
    Use plt to plot train process.

    Args
    - history: history class in keras
    Optional
    - title: title of the figure, default: `Training Performance`
    '''
    # Show loss curves
    plt.figure()
    plt.title(title)
    # val_loss, loss
    plt.plot(history.epoch,
             history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.epoch,
             history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()


def GetIdxPerSNR(snr, nData=4096, nMod=24, nSNR=26):
    ''' 
    Get the data index of input SNR (in Deepsig 2018.01).

    Theory
    - 106496 = nData*nSNR
    - mod n: 106496*n to 106496*(n+1)-1
    - snr n: 4096*n to 4096*(n+1)-1
    '''
    idx = []
    nDataPerSNR = nData * nSNR

    snrId = int((snr + 20)/2)

    for modId in range(nMod):
        # start index of cur mod
        modBaseIdx = nDataPerSNR * modId
        # left index of cur snr in cur mod
        SNRIdxLeft = modBaseIdx + nData * snrId
        # right index of cur snr in cur mod
        SNRIdxRight = modBaseIdx + nData * (snrId+1)
        assert len(list(range(SNRIdxLeft, SNRIdxRight))) == nData
        idx.extend(list(range(SNRIdxLeft, SNRIdxRight)))

    assert len(idx) == nData * nMod
    return idx


def WriteHistory(history, path="history.log"):
    ''' 
    Write train process into a log file.

    Args
    - history: history class in keras
    Optional
    - path: path of the output log file, default: `history.log`
    '''
    with open(path, 'w') as f:
        f.write("\nloss = ")
        f.write(str(history.history['loss']))
        f.write("\nval_loss = ")
        f.write(str(history.history['val_loss']))
        f.write("\naccuracy = ")
        f.write(str(history.history['accuracy']))
        f.write("\nval_accuracy = ")
        f.write(str(history.history['val_accuracy']))
    print("History saved as", path)
