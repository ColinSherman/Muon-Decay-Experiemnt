from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
from pylab import *
from iminuit import Minuit, __version__
print('iminuit version', __version__)

def model(x,b,c):
    return( b*np.exp((1/-c)*x))

def new_model(x,a,b,c,d):
    return((a*np.exp((1/-b)*x))+(c*np.exp((1/-d)*x))+10)


def LSQ(b,c):
    return(sum((qty - model(bins,b,c))**2 / sig**2))



data_x =np.genfromtxt('ourDecays.txt', usecols=(0),unpack=True)

bins = np.linspace(0,4000,201)
qty = np.zeros(201)
sig = np.zeros(201)
for i in data_x:
    qty[int(i/20)] = qty[int(i/20)] + 1

for idx,i in enumerate(sig):
    sig[idx] = np.sqrt(qty[idx])
    # sig[idx] = .5

qty = np.delete(qty,(0,1,2,3,200))
sig = np.delete(sig,(0,1,2,3,200))
bins = np.delete(bins,(0,1,2,3,200))


m = Minuit(LSQ,b=200,c=1111)
m.migrad() # finds minimum of least_squares function
m.hesse()   # computes errors
print(m.values)

plt.plot(bins, qty, color='orange', label="Data")
# plt.yscale('log')
# plt.plot(bins, 40+200*np.exp(bins*(1/-1111)), color='blue')
plt.plot(bins, model(bins, *m.values), color='green', label="Fit", marker='^', markevery= 10)
plt.legend()
plt.xlabel('time (ns)')
plt.ylabel('quantity (log scale)')
plt.title("Decay Rate of Muons")
show()

print(' ')
print(' Fitting parameters')
for p in m.parameters:
    print("{} = {} +- {}".format(p,m.values[p], m.errors[p]))

print(' ')
print('chi square:', m.fval)
print('number of degrees of freedom:', (len(qty)- len(m.parameters)))
print('reduced chi square:', m.fval / (len(qty) - len(m.parameters)) )
