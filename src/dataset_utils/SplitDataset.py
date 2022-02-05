import argparse
import glob
import os
import random
from multiprocessing import Pool, Manager


def argsParse():
    '''Args define
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",
                        help="Data path pattern", default="PicDataset/RadioML2016.10b_filter/")
    parser.add_argument("-o", "--out",
                        help="Output dataset path", default="SpectDataset/RadioML2016.10b_filter/")
    parser.add_argument("-s", "--snr",
                        help="SNR to run, empty for all SNRs", default="")
    parser.add_argument("-r", "--ratio",
                        help="Ratio of train,val,test", default="0.7,0.1,0.2")
    parser.add_argument("-v", "--verbose", type=int,
                        help="Verbose", default=1)
    parser.add_argument("-p", "--process", type=int,
                        help="Process num.", default=12)
    args = parser.parse_args()
    return args


def GetCutSets(full_list, ratio, shuffle=True):
    """Ramdom cut dataset into multi sets
    """

    n_total = len(full_list)
    assert n_total != 0
    if shuffle:
        random.shuffle(full_list)
    n_cut = len(ratio)
    assert n_cut != 0

    offset = [0]
    for i in range(0, n_cut-1):
        offset.append(int(n_total * ratio[i]) + offset[-1])
    offset.append(n_total)
    assert len(offset) == n_cut + 1

    cut_list = []
    for i in range(n_cut):
        cut_list.append(full_list[offset[i]:offset[i+1]])
    assert len(cut_list) == n_cut

    return cut_list


def MakeCutSets(snr):
    class_path_list = glob.glob(os.path.join(data_path, snr, "*"))
    for class_path in class_path_list:
        class_name = class_path.split("/")[-1]
        file_path_list = glob.glob(
            os.path.join(class_path, "*" + FILE_EXT_NAME))
        cut_sets = GetCutSets(file_path_list, cut_ratio)
        for i in range(len(cut_sets)):
            output_folder = os.path.join(os.path.join(
                dataset_path, FOLDER_NAME[i]), snr, class_name)
            os.makedirs(output_folder, exist_ok=True)
            for file_path in cut_sets[i]:
                os.system('cp ' + file_path + ' ' + output_folder)
                if VERBOSE > 0:
                    print(os.path.join(output_folder,
                                       file_path.split("/")[-1]))


def Run(snr_queue, _=-1):
    while not snr_queue.empty():
        MakeCutSets(snr_queue.get())


if __name__ == "__main__":
    args = argsParse()
    data_path = args.data
    dataset_path = args.out
    cut_ratio = list(map(float, args.ratio.split(",")))
    SNR_TO_RUN = args.snr
    VERBOSE = args.verbose
    FILE_EXT_NAME = ".png"
    FOLDER_NAME = ["train", "val", "test"]

    if SNR_TO_RUN:
        print("SNR to run:", SNR_TO_RUN)
        MakeCutSets(SNR_TO_RUN)
    else:

        PROCESS_NUM = args.process
        snr_list = os.listdir(data_path)
        print("Runing all SNRs:", snr_list)
        # init a Manager.Queue for Pool
        snr_queue = Manager().Queue()
        for snr in snr_list:
            snr_queue.put(snr)

        # start multi processing
        pool = Pool(PROCESS_NUM)
        for i in range(PROCESS_NUM):
            pool.apply_async(Run, args=(snr_queue, -1))
        pool.close()
        pool.join()

        print('done')
