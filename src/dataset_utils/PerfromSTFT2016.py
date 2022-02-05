import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy.io
import scipy.signal
import scipy.fftpack
import argparse
from colormap import *
from PIL import Image
from filter import MAG
from multiprocessing import Pool, Manager


def ArgsParse():
    """
    Args define
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-mod",
                        help="Modulation type, default for all.", default='')
    parser.add_argument("-snr", type=int,
                        help="SNR to run, default for all.", default=-1000)
    parser.add_argument("-process", type=int,
                        help="Process num.", default=12)
    parser.add_argument("-path",
                        help="Output dataset path.", default='./PicDataset/RadioML2016.10b_filter')
    parser.add_argument("-dat",
                        help="Dataset .dat path.", default='./OriginalDataset/RML2016.10b.dat')
    parser.add_argument("-filter", action="store_true",
                        help="Whether to use Gaussian Filter.", default=True)
    args = parser.parse_args()
    return args


def Mkdir(path):
    ''' 
    Make dirs
    '''
    os.makedirs(path, exist_ok=True)


def Data2Pic(dat, snr, mod, sigID, out_dataset_path, use_filter):
    # pic save path
    folder = out_dataset_path + '/' + str(snr) + "/" + mod + "/"
    Mkdir(folder)
    filePath = folder + str(sigID) + '.png'

    I = dat[0]
    Q = dat[1]
    signal = I + 1j*Q
    if use_filter:
        denoise_signal = np.array(MAG(signal), dtype=complex)
        signal = denoise_signal
    R = 38
    window_length = 40
    _, _, S = scipy.signal.stft(signal,
                                fs=1.0,
                                window='hamming',
                                nperseg=window_length,
                                noverlap=R,
                                nfft=256,
                                boundary=None,
                                detrend=False,
                                return_onesided=False
                                )
    spec = 20 * np.log10(np.abs(S)+2.2204e-16)
    plt.matshow(spec, cmap=parula())
    plt.axis('off')
    plt.savefig(filePath, bbox_inches='tight', pad_inches=0.0)
    plt.close()
    img = Image.open(filePath)
    img = img.resize((256, 256))
    img.save(filePath)


def Run(queue, out_dataset_path, use_filter=False):
    while not queue.empty():
        mod, snr = queue.get()
        data = np.array(loaded_signals[(mod, snr)])
        for i in range(data.shape[0]):
            # current one signal: loaded_signals[(mod, snr)][i, ::]
            print('Main: Mod', mod, ' SNR', snr, ' > ',
                  i + 1, '/', data.shape[0])
            Data2Pic(data[i, ::], snr, mod, i + 1,
                     out_dataset_path, use_filter)


if __name__ == "__main__":
    # Get args
    args = ArgsParse()

    MOD = args.mod
    SNR = args.snr
    out_dataset_path = args.path
    datFile = args.dat
    use_filter = args.filter
    process_num = args.process

    print(".dat File:", datFile)
    if MOD:
        print("Mod To run:", MOD)
    if SNR != -1000:
        print("SNR To run:", SNR)
    print("Output dataset:", out_dataset_path)
    print("Use Gaussian Filter:", use_filter)

    """
        Main
    """
    # load data from *.dat
    loaded_signals = pickle.load(open(datFile, 'rb'), encoding='latin-1')
    snrs, mods = map(lambda j: sorted(
        list(set(map(lambda x: x[j], loaded_signals.keys())))), [1, 0])
    print('Avaliable SNRs:', snrs)
    print('Avaliable Mods:', mods)

    # init a Manager.Queue for Pool
    signal_key_queue = Manager().Queue()
    for key in loaded_signals.keys():
        print(key)
        signal_key_queue.put(key)

    # start multi processing
    pool = Pool(process_num)
    for i in range(process_num):
        pool.apply_async(Run, args=(
            signal_key_queue, out_dataset_path, use_filter))
    pool.close()
    pool.join()

    print('done')
