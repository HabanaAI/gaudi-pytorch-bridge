# parse log file and generate various plots
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import re
import pandas as pd
import matplotlib
matplotlib.use('agg')

device_list = []
# process for each log file


def process(args):
    for file in args.files:
        with open(file.name) as f:
            vl = []
            b1 = {}
            bd = {}
            temp = 0
            for line in f:
                if temp == 2:
                    headers = bd.keys()
                    vl.append(bd)
                    bd = {}
                    temp = 0
                if 'Namespace' in line:
                    m = line[line.find("(") + 1:line.find(")")].split(', ')
                    for i in m:
                        try:
                            x = i.split('=')
                            b1[x[0].strip()] = x[1].strip()
                        except:
                            pass
                if 'Epoch=' in line:
                    temp = temp + 1
                    m = re.split('([^ =  ]  )', line)
                    for i in m:
                        try:
                            x = i.split('=')
                            if x[0].strip() == 'lossy':
                                bd[x[0].strip()] = x[1].strip()
                        except:
                            pass
                if 'Epoch:' in line:
                    temp = temp + 1
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
        device_list.append((re.sub('[\W\_]', '', b1['device']), '( Model: ' +
                            re.sub('[\W\_]', '', b1['model']) + '; batchsize: ' + b1['batch_size'] + ' )'))

        # writing data into respective csv file
        with open(re.sub('[\W\_]', '', b1['device']) + ".csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(vl)


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
    process(args)

    # making plot for each device
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)

    for device in device_list:
        dataframe = pd.read_csv(device[0] + '.csv')
        dataframe = dataframe.dropna()
        x = pd.to_numeric(dataframe.Epoch)
        y = pd.to_numeric(dataframe.loss_avg)
        w = pd.to_numeric(dataframe.acc1_avg)
        z = pd.to_numeric(dataframe.acc5_avg)
        plt.figure(1)
        plt.plot(y, label='id %s' % y)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.legend([i[0] for i in device_list], loc="upper right")
        tick_values = list(CountFrequency(x.values.tolist()).values())
        xaxis_values = tuple(set(x))
        plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
        plt.grid(True, linestyle='dashed')
        plt.title(y.name + ' ' + device[1])
        plt.savefig(x.name + '_' + y.name + '_plot.png', dpi=300)
        plt.figure(2)
        plt.plot(w, label='id %s' % w)
        plt.xlabel(x.name)
        plt.ylabel(w.name)
        plt.legend([i[0] for i in device_list], loc="lower right")
        tick_values = list(CountFrequency(x.values.tolist()).values())
        xaxis_values = tuple(set(x))
        plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
        plt.grid(True, linestyle='dashed')
        plt.title(w.name + ' ' + device[1])
        plt.savefig(x.name + '_' + w.name + '_plot.png', dpi=300)
        plt.figure(3)
        plt.plot(z, label='id %s' % z)
        plt.xlabel(x.name)
        plt.ylabel(z.name)
        plt.legend([i[0] for i in device_list], loc="lower right")
        tick_values = list(CountFrequency(x.values.tolist()).values())
        xaxis_values = tuple(set(x))
        plt.xticks(np.array(CumulativeSum(tick_values)), xaxis_values)
        plt.grid(True, linestyle='dashed')
        plt.title(z.name + ' ' + device[1])
        plt.savefig(x.name + '_' + z.name + '_plot.png', dpi=300)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


if __name__ == '__main__':
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+', help='process only available files')
    args = parser.parse_args()
    main(args)
