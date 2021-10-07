import numpy as np
import sys
import argparse
from glob import glob


parser = argparse.ArgumentParser(description='Converts from .data to .txt and does some filtering of the data')
parser.add_argument('-f', '--file',
                    default="Data/*.data",
                    type=str,
                    help='data input file')
parser.add_argument('-o', '--output',
                    default="Data/test.txt",
                    type=str,
                    help='data output file')

params = parser.parse_args()
data_x = []
data_y = []
print(glob('*.data'))
for filename in glob('*.data'):
    x,y = np.genfromtxt(filename, usecols=(0,1),unpack=True)
    data_x.extend(x)
    data_y.extend(y)
new = list(zip(data_x,data_y))
decays = [x for x in new if not x[0] >= 40000]

np.savetxt(params.output,decays,fmt='%2i')
