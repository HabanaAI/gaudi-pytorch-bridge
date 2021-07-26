#parse log file and generate various plots. Parameters  <file1 file2 ..>
import argparse
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

device_list = []
#process for each log file
def process(args):
    dflist = []
    b1 = {}
    device_name2='cuda_'
    data_type=" FP32 "
    tta_type=""
    deep_s=""
    for file in args.files:
        with open(file.name,'rb') as f:
            v1 = []
            for bline  in f:
                try:
                    line = bline.decode()
                    #print(line)
                except UnicodeDecodeError as e:
                    pass
                v2=[]
                if 'loss=' in line and 'mean' in line:
                    strip_l= line[line.find('loss'):].split(']')

                    strip_l[1] = strip_l[1][strip_l[1].find('Epoch:'):]
                    strip_l[1] = strip_l[1].replace("mean dice","mean_dice")
                    strip_l[1] = strip_l[1].rstrip()
                    strip_l[1] = strip_l[1].replace(' : ',':').replace(': ',':')
                    strip_l[1] = strip_l[1].replace('  ',' ')
                    split2 = strip_l[1].split(' ')
                    for i in split2:
                        v2.append(float(i.split(':')[1]))

                    v2.append(float(strip_l[0].split('=')[1]))
                    v1.append(v2)
                else:
                    if 'Namespace' in line:
                        m = line[line.find("(")+1:line.find(")")].split(', ')
                        for i in m:
                            try:
                                x = i.split('=')
                                b1[x[0].strip()] = x[1].strip()
                            except:
                                pass
                    continue;

            head_list = ['Epoch', 'mean_dice','TOP_mean','L1','L2','L3','TOP_L1','TOP_L2','TOP_L3','val_loss','loss']
            head_list2=[]
            if  b1.get('device') is not None:
                device_name2 = b1['device'].replace("'","") + "_"
            for i in range(0,len(head_list)):
                test = device_name2 + head_list[i]
                head_list2.append(test)

            df = pd.DataFrame(v1, columns = head_list2)
            df = df.drop([head_list2[0]], axis=1)
            dflist.append(df)

    fig, axs = plt.subplots(figsize=(10, 10))
    plt.legend(bbox_to_anchor=(1.0, 1.0))

    if b1['is_hmp'] == 'True':
        data_type = ' BF16 '

    if b1['tta'] == 'True':
        tta_type = " tta "

    if b1['deep_supervision'] == 'True':
        deep_s = " deep_supervision "

    title = tta_type + data_type + deep_s + "BS_" + b1['batch_size']
    y = None
    color_list=['red', 'green']
    j=0
    for frame in dflist:
        if y is None:
            y = frame.plot(subplots=True, figsize=(20, 20), color=color_list[j],ax=axs, title=title)
        else:
            y = frame.plot(subplots=True, figsize=(20, 20), color=color_list[j], ax=y, title=title)
        j = j + 1
        if j >= len(color_list):
            j = 0

    fig.savefig("unet_plot.png")



def main(args):
    process(args)

if __name__ == '__main__':
    #for command line arguments
    parser = argparse.ArgumentParser()
        #TODO: add argument parser to compare the files wrt epoch/device type etc.
    parser.add_argument('files', type=argparse.FileType('r'), nargs='+', help='process only available files')
    args = parser.parse_args()
    main(args)
