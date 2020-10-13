import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
import math

def get_values(filterKey, _cards, log):
    filtered_values = []
    for line in open(log):
        if line.find("Finished training it") != -1:
            words = line.split()
            idx = words.index(filterKey)
            if filterKey == 'ms/it,':
                filtered_values.append(float(words[idx-1].rstrip(" ")))
            else:
                filtered_values.append(float(words[idx+1].rstrip(",")))

    averaged_values = []
    averaged_values = [round((sum(filtered_values[i:i+_cards]) / _cards), 6)
        for i in range(0, len(filtered_values), _cards)]

    #print(averaged_values)
    return averaged_values

def main(argv):
    try:
        inLog1 = sys.argv[1] #single chip log
        inLog2 = sys.argv[2] #multichip log
        #loss, accuracy, time
        filterKey = sys.argv[3]
        cards = int(sys.argv[4])
    except:
        print(" USAGE: python get_scaling.py 'singlecard.log' 'multicard.log' 'key_to_be_parsed' 'num_cards' \n\
 EXAMPLE: python get_scaling.py sc.txt mc.txt 'ms/it,' 2")
        sys.exit(1)

    import statistics
    sc_average = statistics.mean(get_values(filterKey, 1, inLog1))
    mc_average = statistics.mean(get_values(filterKey, cards, inLog2))
    print("Mean of ", filterKey, " in ", inLog1, " is : ", sc_average)
    print("Mean of ", filterKey, " in ", inLog2, " is : ", mc_average)
    scaling = (sc_average/mc_average)*100
    print("scaling = ", scaling)

if __name__ == "__main__":
   main(sys.argv[1:])