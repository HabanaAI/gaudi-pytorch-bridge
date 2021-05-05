# parse log file and generate various plots
import numpy as np
import argparse
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

# Parse all the log files and create some meta data like device, model, batchsize etc
# if there are multiple log files of same device, append the device name with an order number.
def ParseHeader(args):
    log_info_list = []
    dev_array =[]
    for file in args.files:
        with open(file.name) as f:
            log_info_dict = {}
            for line in f:
                if 'Namespace' in line:
                    line = line.split('Namespace(',1)[1]

                    m = line.split(', ')
                    for i in m:
                        try:
                            x = i.split('=')
                            log_info_dict[x[0].strip()] = x[1].strip()
                        except:
                            pass

                    device='cpu'

                    habana_key = 'use_habana'
                    gpu_key = 'no_cuda'
                    if log_info_dict[habana_key] == 'True':
                        device = 'hpu'
                    elif log_info_dict[gpu_key] == 'False':
                        device = 'gpu'

                    log_info = [
                        device,
                        '( BERT -- ' +
                            ' train_batchsize : ' + log_info_dict['per_gpu_train_batch_size'] + ' )']

                    log_info_list.append(log_info)

                    dev_array.append(log_info[0])
                    break

    # if there are multiple log files of same device, append the device name with an order number.
    # If log files f1, f2, f3 correspond to devices 'habana', 'cpu', habana', change deice names as
    # 'habana1', 'cpu', 'habana2'. This is to prevent overwriting

    dev_count = {}
    for dev in dev_array:
        dev_count[dev] = dev_array.count(dev)

    for dev, count in dev_count.items():
        if count > 1:
            i = 1
            for lg_info in log_info_list:
                if lg_info[0] == dev:
                    lg_info[0] = dev + '_' + str(i)
                    i = i +1
    return log_info_list

def Process(args):
    log_info_list = ParseHeader(args)

    k = 0
    out_file_list = []
    for file in args.files:
        ldict_list = []
        with open(file.name) as f:
            ep_str = 'Epoch'
            ep_val = 0

            for line in f:
                l_dict = {}
                if 'Epoch:' in line:
                    tokens = line.split('|')
                    for token in tokens:
                        if 'it' in token:
                            ep_val = token.split('/')[0]

                l_dict[ep_str] = ep_val

                if 'Loss' in line and 'eval_loss' not in line:
                   l_dict['Loss'] = line.split('Loss: ',1)[1]
                   ldict_list.append(l_dict)

        # writing data into respective csv file
        out_file = os.path.join(args.out_dir, log_info_list[k][0] + '.csv')
        k = k + 1

        # ldict_list is an array of dicts of parsed log entries
        # use the first entry to get the header
        headers = ldict_list[0].keys()
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(ldict_list)
        out_file_list.append(out_file)

    return log_info_list, out_file_list

def CountFrequency(my_list):
    freq = {}
    for items in my_list:
        freq[items] = my_list.count(items)
    return freq


def CumulativeSum(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length + 1)]
    return cu_list

def CreatePng(args, log_info_list, out_file_list):
    fig1 = plt.figure(1)
    i = 0
    suffix = ''
    for log_info in log_info_list:
        out_file = out_file_list[i]
        i = i + 1
        dataframe = pd.read_csv(out_file)
        dataframe = dataframe.dropna()

        x = pd.to_numeric(dataframe.Epoch)
        y = pd.to_numeric(dataframe.Loss)

        plt.figure(1)
        plt.plot(y, label='id %s' % y)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.legend([i[0] for i in log_info_list], loc="upper right")
        tick_values = list(CountFrequency(x.values.tolist()).values())
        xaxis_values = tuple(set(x))
        plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
        plt.grid(True, linestyle='dashed')
        plt.title(y.name + ' ' + log_info[1])
        suffix += '_' + log_info[0]
        plt.savefig(os.path.join(args.out_dir, y.name + suffix + '.png'), dpi=300)

    plt.close(fig1)
    return

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    log_info_list, out_file_list = Process(args)
    print('log_info_list : {}'.format(log_info_list))
    CreatePng(args, log_info_list, out_file_list)
    return

if __name__ == '__main__':
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='out_csv_plot',
                        help='Path including name of dir to place outputs like plot and csv files')
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+',
                        help='files to parse. Multiple files can be given')
    args = parser.parse_args()
    main(args)
