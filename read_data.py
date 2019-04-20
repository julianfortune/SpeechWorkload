'''
Created on Jul 10, 2017

@author: Jamison Heard
'''

import os
import tflearn.data_utils as data
from tflearn.data_utils import load_csv
import numpy as np

def import_file(file):
    data,label = load_csv(file, has_header = True)
    return data

def parse_time(time_string):
    minute_string = time_string[0:2]
    second_string = time_string[3:]
    seconds = int(minute_string) * 60 + int(round(float(second_string), 0))
    return seconds


def main():
    file_name = "./actual/normal_load.csv"
    data = import_file(file_name)
    output = np.empty(0)
    current_level = 0
    for index in range(0,len(data)):
        row = data[index]
        current_time = parse_time(row[0])
        current_level = int(row[10])
        np.append(output, current_level)
        if index + 1 < len(data):
            next_time = parse_time(data[index + 1][0])
            for interpolated_time in range(current_time + 1, next_time):
                output = np.append(output, current_level)

    print(output)

    np.save("./labels/" + os.path.basename(file_name)[:-4],output)

if __name__ == "__main__":
    main()
