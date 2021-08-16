# imports
from typing import TypeVar
import numpy as np
from matplotlib import pyplot as plt
import math as math

def ode_model_pressure(t, P, b, Pa, Pmar):
    #change pressure parameter depending on time that MAR is introduced
    tmar = 2010
    if t < tmar:
        Pa1 = Pa
    else:
        Pa1 = Pa + Pmar
    
    dPdt = -b * (P + (Pa/2)) - (b * (P - (Pa1/2)))
    return dPdt

def ode_model_concentration(t, M, P, tdelay, a, P0, C, bc, Pa, Pmar, b):

    #change pressure parameter depending on time that MAR is introduced
    tmar = 2010
    if t < tmar:
        Pa1 = Pa
    else:
        Pa1 = Pa + Pmar

    #change infiltration depending on time that active carbon introduced
    tc = 2010
    if t < tc:
        b = ode_model_pressure(t-tdelay, P, b, Pa, Pmar)
    else:
        b = a * ode_model_pressure(t-tdelay, P, b, Pa, Pmar)

    n = stock_interpolation(t-tdelay)
    
    dCdt = (((-n) * b * (P-P0))/M) + ((bc * (P - (Pa1/2)) * C)/M)
    return dCdt


def stock_population():
    ''' Returns year and stock population from data

        Parameters:
        -----------
        none

        Returns:
        --------
        year :  array-like
                Vector of times (years) at which measurements were taken.
        stock : array-like
                Vector of stock populations
    '''
    
    year = np.genfromtxt('nl_cows.txt', delimiter = ',', skip_header = 1, usecols = 0)
    stock = np.genfromtxt('nl_cows.txt', delimiter = ',', skip_header = 1, usecols = 1)

    return year, stock

def stock_interpolation(t):
    ''' Return stock parameter n for model

        Parameters:
        -----------
        t : array-like
            Vector of times at which to interpolate the stock population

        Returns:
        --------
        n : array-like
            Stock population interpolated at t.
    '''

    year, stock = stock_population()

    n = np.interp(t, year, stock)

    return n


def improved_euler_concentration(f, t0, t1, dt, x0, pars):
    
    # Allocate return arrays
    t = np.arange(t0, t1+dt, dt)
    params_unknown, params_known = pars
    x = np.zeros(len(t))
    x[0] = x0

    for i in range(0, (len(t) - 1)):
        
        # Compute normal euler step
        x1 = x[i] + dt*f(t[i], x[i], params_unknown, params_known,i)
        
        # Corrector step
        x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known,i) + f(t[i+1], x1, params_unknown, params_known,i))

    return x

def plot_given_data():
    year_stock = np.genfromtxt("nl_cows.txt", delimiter = ',', skip_header = 1, usecols = 0)
    stock = np.genfromtxt('nl_cows.txt', delimiter = ',', skip_header = 1, usecols = 1)

    year_conc = np.genfromtxt("nl_n.csv", delimiter = ',', skip_header = 1, usecols = 0)
    concentration = np.genfromtxt('nl_n.csv', delimiter = ',', skip_header = 1, usecols = 1)

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.plot(year_conc, concentration, label = "Concentration", color = 'red')

    ax2 = ax1.twinx()
    ax2.set_xlabel("time, [years]")
    ax2.set_ylabel("Stock numbers")
    stck = ax2.plot(year_stock, stock, label = "Stock numbers", color = 'green')
    lns = conc+stck
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    fig.tight_layout()

    plt.annotate(xy=[2010,250000], s='  MAR introduced')
    plt.plot([2010,2010], [0,700000], color =  'black', linestyle = 'dashed')

    plt.show() 

#####################################################################################################
#-functions for odes (pressure and concentration)
#-function for reading in stock data
#-function for reading in concentration data
#-function for interpolating stock numbers
#-function for solving ode model using improved eulers
#-plot the LPM over the top of the data
#-calibrate
#-compare analytical and numerical plots with a benchmarking function
#-forecast
#########################################################################################################
#testing git 
#testing
#trent was here


if __name__ == "__main__":
    #ode_model_pressure()
    #ode_model_concentration()
    stock_population()


