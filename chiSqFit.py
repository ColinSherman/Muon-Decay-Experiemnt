from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
from pylab import *
from iminuit import Minuit, __version__
print('iminuit version', __version__)

def model(x,a,b,c):
    return(a + b*np.exp(-c*x))

data_x, data_y, ysig = np.genfromtxt('chisqFit_exponentialData.txt', usecols=(0,1,2),unpack=True)

def LSQ(a,b,c):
    return(sum((data_y - model(data_x,a,b,c))**2 / ysig**2))

m = Minuit(LSQ,a=1,b=1,c=1)
m.migrad() # finds minimum of least_squares function
m.hesse()   # computes errors

plt.plot(data_x, model(data_x, *m.values), color='orange')
show()
