import numpy as np
import pandas
import os

def read_txt(file_name):

    data = np.loadtxt(file_name, dtype=np.str_, delimiter=None, unpack=False)
    print(data)
    print(data.shape)
    data = np.delete(data, 0, 0)
    data = np.delete(data, 0, 1)
    print("-" * 50)
    print(data)
    print(data.shape)

    return data


def main():

    data1 = read_txt('1.txt')
    data2 = read_txt('2.txt')

    result = np.correlate(data1, data2, mode='full')

    print(result)


if __name__ == "__main__":
    main()