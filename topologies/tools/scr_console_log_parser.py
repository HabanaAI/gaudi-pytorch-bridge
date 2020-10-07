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
    def parse_line(line):
        bd = {}
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
        return bd
    def write_csv_file(out_file, vl):
        # writing data into respective csv file
        #out_file = os.path.join(args.out_dir, dev + '.csv')
        #k = k + 1
        headers = vl[0].keys()# vl is an array of dicts of parsed log entries. use the first entry to get the header
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(vl)

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
    out_file_list_t = []
    out_file_list_v = []
    k =0
    for file in args.files:
        vl_t = [] # for training
        vl_v = [] # for validation
        epoch = 0
        with open(file.name) as f:
            for line in f:
                if 'Epoch:' in line and 'eta' in line:
                    bd = parse_line(line)
                    vl_t.append(bd)
                    epoch = bd['Epoch'] # Store epoch value for printing during eval/test log parse
                if 'Test' in line and 'eta' in line:
                    bd = parse_line(line)
                    bd['Epoch'] = epoch # Test logs dont have epoch. so use the epoch number from train
                    vl_v.append(bd)


        # writing data into respective csv file
        out_file_t = os.path.join(args.out_dir, log_info_list[k][0] + '_train.csv')
        write_csv_file(out_file_t, vl_t)
        out_file_list_t.append(out_file_t)

        out_file_v = os.path.join(args.out_dir, log_info_list[k][0] + '_val.csv')
        write_csv_file(out_file_v, vl_v)
        out_file_list_v.append(out_file_v)

    return out_file_list_t, out_file_list_v



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

    def plot_graph(out_file_list, tag):
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        fig3 = plt.figure(3)
        tag_l = tag.lower()

        i = 0
        for log_info in log_info_list:
            out_file = out_file_list[i]
            i = i +1
            dataframe = pd.read_csv(out_file)
            #dataframe = dataframe.dropna()
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
            plt.title(tag + ' : '+ y.name + ' ' + log_info[1])
            plt.savefig(os.path.join(args.out_dir, y.name + '_plot_' + tag_l + '.png'), dpi=300)
            plt.figure(2)
            plt.plot(w, label='id %s' % w)
            plt.xlabel(x.name)
            plt.ylabel(w.name)
            plt.legend([i[0] for i in log_info_list], loc="lower right")
            tick_values = list(CountFrequency(x.values.tolist()).values())
            xaxis_values = tuple(set(x))
            plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
            plt.grid(True, linestyle='dashed')
            plt.title(tag + ' : '+ w.name + ' ' + log_info[1])
            plt.savefig(os.path.join(args.out_dir, w.name + '_plot_' + tag_l + '.png'), dpi=300)
            plt.figure(3)
            plt.plot(z, label='id %s' % z)
            plt.xlabel(x.name)
            plt.ylabel(z.name)
            plt.legend([i[0] for i in log_info_list], loc="lower right")
            tick_values = list(CountFrequency(x.values.tolist()).values())
            xaxis_values = tuple(set(x))
            plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
            plt.grid(True, linestyle='dashed')
            plt.title(tag + ' : '+ z.name + ' ' + log_info[1])
            plt.savefig(os.path.join(args.out_dir, z.name + '_plot_' + tag_l + '.png'), dpi=300)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file_list_t, out_file_list_v = process(args)
    plot_graph(out_file_list_t, 'Training')
    plot_graph(out_file_list_v, 'Validation')


if __name__ == '__main__':
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='console_log_parser_output',
                        help='Path including name of dir to place outputs like plot and csv files')
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+',
                        help='files to parse. Multiple files can be given')
    args = parser.parse_args()
    main(args)
