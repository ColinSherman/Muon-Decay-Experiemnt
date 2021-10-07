import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Converts from .data to .txt and does some filtering of the data')
parser.add_argument('-f', '--file',
                    default="ourData.txt",
                    type=str,
                    help='data input file')
params = parser.parse_args()

x = np.genfromtxt(params.file, usecols=(1),unpack=True)
num_bins = 400

n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5, ec = 'black')
plt.show()
