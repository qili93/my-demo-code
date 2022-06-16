# usage: python get_time.py xxx.log
# output: 
# time cost (min):  0:07:12
# time cost (sec):  432.0

import sys
import datetime


def get_data(file_name):
    global start_time, end_time
    with open(file_name, 'r') as f:
        for line in f:
            if line.find("Start Time is: ") != -1:
                start_time = line.strip().split("is: ")[1].strip()
            if line.find("End Time is: ") != -1:
                end_time = line.strip().split("is: ")[1].strip()
        return start_time, end_time


if __name__ == '__main__':
    start1, end1 = get_data(sys.argv[1])

    start_time = datetime.datetime.strptime(start1, "%m/%d/%Y %H:%M:%S")
    end_time = datetime.datetime.strptime(end1, "%m/%d/%Y %H:%M:%S")
    spend_time = end_time - start_time

    print("time cost (min): ", spend_time)
    print("time cost (sec): ", spend_time.total_seconds())

