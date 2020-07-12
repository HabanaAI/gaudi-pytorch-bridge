# parse log file and generate various plots
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import re
import os
import pandas as pd
import matplotlib
matplotlib.use('agg')

log_info_list = []

def process(args):
    # Parse all the log files and create some meta data like device, model, batchsize etc
    # if there are multiple log files of same device, append the device name with an order number.
    def log_meta_data(args):
        dev_array =[]
        for file in args.files:
            with open(file.name) as f:
                b1 = {}
                for line in f:
                    if 'Namespace' in line:
                        m = line[line.find("(") + 1:line.find(")")].split(', ')
                        for i in m:
                            try:
                                x = i.split('=')
                                b1[x[0].strip()] = x[1].strip()
                            except:
                                pass
                        log_info = [re.sub('[\W\_]', '', b1['device']), '( Model: ' +
                                re.sub('[\W\_]', '', b1['model']) + '; batchsize: ' + b1['batch_size'] + ' )']
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
                        lg_info[0] = dev + str(i)
                        i = i +1

    #=============================def log_meta_data(args) ends ==================================================

    log_meta_data(args)
    out_file_list = []
    k =0
    for file in args.files:
        vl = []
        with open(file.name) as f:
            for line in f:
                bd = {}
                if 'Epoch:' in line and 'eta' in line:
                    if 'Test' in line: # TODO add handling of testi/val iteration parsing
                        continue
                    else:
                        m = line.split('  ')
                        for i in m:
                            try:
                                x = i.split(':')
                                if x[0].strip() in ['loss', 'acc1', 'acc5']:
                                    y = x[1].strip().split(' ')
                                    bd[x[0].strip() + '_median'] = y[0].strip()
                                    bd[x[0].strip() + '_avg'] = re.sub('[\\(|\\)]', '', y[1].strip())
                                else:
                                    bd[x[0].strip()] = re.sub('[\\[|\\]]', '', x[1].strip())
                            except:
                                pass
                    vl.append(bd)

        # writing data into respective csv file
        out_file = os.path.join(args.out_dir, log_info_list[k][0] + '.csv')
        k = k + 1
        headers = vl[0].keys()# vl is an array of dicts of parsed log entries. use the first entry to get the header
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(vl)
        out_file_list.append(out_file)
    return out_file_list



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


def main(args):

    os.makedirs(args.out_dir, exist_ok=True)
    out_file_list = process(args)

    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)

    i = 0
    for log_info in log_info_list:
        out_file = out_file_list[i]
        i = i +1
        dataframe = pd.read_csv(out_file)
        dataframe = dataframe.dropna()
        x = pd.to_numeric(dataframe.Epoch)
        y = pd.to_numeric(dataframe.loss_avg)
        w = pd.to_numeric(dataframe.acc1_avg)
        z = pd.to_numeric(dataframe.acc5_avg)
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
        plt.savefig(os.path.join(args.out_dir, x.name + '_' + y.name + '_plot.png'), dpi=300)
        plt.figure(2)
        plt.plot(w, label='id %s' % w)
        plt.xlabel(x.name)
        plt.ylabel(w.name)
        plt.legend([i[0] for i in log_info_list], loc="lower right")
        tick_values = list(CountFrequency(x.values.tolist()).values())
        xaxis_values = tuple(set(x))
        plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
        plt.grid(True, linestyle='dashed')
        plt.title(w.name + ' ' + log_info[1])
        plt.savefig(os.path.join(args.out_dir, x.name + '_' + w.name + '_plot.png'), dpi=300)
        plt.figure(3)
        plt.plot(z, label='id %s' % z)
        plt.xlabel(x.name)
        plt.ylabel(z.name)
        plt.legend([i[0] for i in log_info_list], loc="lower right")
        tick_values = list(CountFrequency(x.values.tolist()).values())
        xaxis_values = tuple(set(x))
        plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
        plt.grid(True, linestyle='dashed')
        plt.title(z.name + ' ' + log_info[1])
        plt.savefig(os.path.join(args.out_dir, x.name + '_' + z.name + '_plot.png'), dpi=300)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


if __name__ == '__main__':
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='console_log_parser_output',
                        help='Path including name of dir to place outputs like plot and csv files')
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+',
                        help='files to parse. Multiple files can be given')
    args = parser.parse_args()
    main(args)
