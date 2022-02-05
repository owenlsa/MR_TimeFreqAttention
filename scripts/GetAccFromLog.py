import os
import argparse
import re
from glob import glob


def ArgsParse():
    """
        Args define
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset',
                        help="Dataset to run.", default='RadioML2016.10b')
    parser.add_argument('-p', '--path',
                        help="Saved log path.", default='./saved_models/')
    parser.add_argument('-m', '--method',
                        help="Used method.", default='DSF_SE_SC')
    parser.add_argument("-v", "--verbose", type=int,
                        help="Verbose, 0 for Silent, 1 for Info, 2 for Debug", default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = ArgsParse()
    saved_log_path = args.path
    method = args.method
    dataset = args.dataset
    snrs = list(range(18, -22, -2))
    accs = dict.fromkeys(snrs, 0.0)
    models = dict.fromkeys(snrs, '')

    log_path_list = glob(os.path.join(
        saved_log_path, dataset + "-*dB-" + method + "-*.log"))
    assert len(log_path_list) != 0
    for log_path in log_path_list:
        try:
            if args.verbose > 1:
                print(log_path)
            with open(log_path, "r") as f:
                last_line = f.readlines()[-1].strip()
        except:
            if args.verbose > 1:
                print("skip an error file:", log_path)
            continue
        if "Acc:" in last_line:
            nums = re.findall(r'-?\d+\.?\d*e?-?\d*?', last_line)
            assert len(nums) == 2, "error finding SNR and acc in log"
            cur_snr = int(nums[0])
            cur_acc = float(nums[1])
            if cur_acc > accs[cur_snr]:
                accs[cur_snr] = cur_acc
                models[cur_snr] = log_path

    for k in accs.keys():
        print("SNR: %d, acc: %.3f, model: %s" %
              (k, accs[k], models[k]))
