from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from pprint import pprint

from iminuit import Minuit, __version__
print('iminuit version', __version__)

data_x, data_y, ysig = genfromtxt('chisqFit_exponentialData.txt', usecols=(0,1,2),unpack=True)
errorbar(data_x,data_y,yerr=ysig,fmt="o",color='k',solid_capstyle='projecting',capsize=5)

def model(x, a, b, c):
    return a + b*np.exp(-c*x)

def least_squares(a,b,c):
    return sum((data_y - model(data_x,a,b,c))**2 / ysig**2)

m = Minuit(least_squares, a=1, b=2, c=2)
m.migrad() # finds minimum of least_squares function
m.hesse() # computes errors
print(' ')
#print(' Covariance matrix')
# print(m.matrix())
print(' ')
plt.plot(data_x, model(data_x, *m.values.values()), color='orange')

print(' ')
print(' Fitting parameters')
for p in m.parameters:
    print("{} = {} +- {}".format(p,m.values[p], m.errors[p]))

print(' ')
print('chi square:', m.fval)
print('number of degrees of freedom:', (len(data_y)- len(m.parameters)))
print('reduced chi square:', m.fval / (len(data_y) - len(m.parameters)) )

show()
