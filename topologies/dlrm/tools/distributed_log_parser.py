import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import math

def get_filtered_values(filterKey, _cards, log):
    filtered_values = []
    for line in open(log):
        if line.find("Finished training it") != -1:
            words = line.split()
            idx = words.index(filterKey)
            filtered_values.append(float(words[idx+1].rstrip(",")))

    averaged_values = []
    averaged_values = [round((sum(filtered_values[i:i+_cards]) / _cards), 6)
        for i in range(0, len(filtered_values), _cards)]

    #print(averaged_values)
    return averaged_values

def main(argv):
    try:
        mcLog = sys.argv[1]
        scLog = sys.argv[2]
        #loss, accuracy
        filterKey = sys.argv[3]
        Ylabel = str(filterKey)
        Xlabel = 'iterations'
        cards = int(sys.argv[4])
    except:
        print(" USAGE: python distributed_log_parser.py 'multi_chip.log' 'single_chip.log' 'key_to_be_parsed' 'num_cards' \n\
 EXAMPLE: python distributed_log_parser.py mctrain.txt sctrain.txt 'accuracy' 2 \n This will generate 'dlrm_plot.png' with the plots")
        sys.exit(1)

    plt.plot(get_filtered_values(filterKey, cards, mcLog), label=str(cards)+'-chip')
    plt.plot(get_filtered_values(filterKey, 1, scLog), label='single-chip')
    plt.ylabel(Ylabel)
    plt.xlabel(Xlabel)
    plt.legend()
    plt.savefig("dlrm_plot.png")

if __name__ == "__main__":
   main(sys.argv[1:])