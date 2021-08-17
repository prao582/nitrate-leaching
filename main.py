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
    
    year_stock = np.genfromtxt('nl_cows.txt', delimiter = ',', skip_header = 1, usecols = 0)
    stock = np.genfromtxt('nl_cows.txt', delimiter = ',', skip_header = 1, usecols = 1)

    return year_stock, stock

def nitrate_concentration():
    ''' Returns nitrate concentration from data

        Parameters:
        -----------
        none

        Returns:
        --------
        year :          array-like
                        Vector of times (years) at which measurements were taken.
        concentration : array-like
                        Vector of nitrate concentrations 
    '''

    year_conc = np.genfromtxt("nl_n.csv", delimiter = ',', skip_header = 1, usecols = 0)
    concentration = np.genfromtxt('nl_n.csv', delimiter = ',', skip_header = 1, usecols = 1)

    return year_conc, concentration

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


def improved_euler_pressure(f,t0, t1, dt, x0, pars):
    """Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time value
        t1 : float
            Final time value
        dt : float
            Time step used
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. Parameters that will later be passed to the curve fitting function
            4. all other parameters
            Refer to the function definitions for order within (3) and (4)
            5. Optional - counter number to be passed to array parameters
    """

    # Allocate return arrays
    t = np.arange(t0, t1+dt, dt)
    params_unknown, params_known = pars
    x = np.zeros(len(t))
    x[0] = x0
    i=0
    # Check if q is iterable or const
    if isinstance(params_known[0], float) == True:
        dqdt = 0
        if len(params_known) != 2:
            params_known.append(dqdt)
        # Loop through time values, finding corresponding x value
        for i in range(0, (len(t) - 1)):
            # Compute normal euler step
            x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known)
            # Corrector step
            x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known) + f(t[i+1], x_temp, params_unknown, params_known))
    else:
        # Get dqdt and append to known parameters
        dqdt = np.gradient(params_known[0])
        if len(params_known) != 2:
            params_known.append(dqdt)
        # Loop through time values, finding corresponding x value
        for i in range(0, (len(t) - 1)):
            # Compute normal euler step
            x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known, i=i)
            # Corrector step
            x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known, i=i) + f(t[i+1], x_temp, params_unknown, params_known, i=i))
        
    return x

def plot_given_data():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')

    ax2 = ax1.twinx()
    ax2.set_xlabel("time, [years]")
    ax2.set_ylabel("Stock numbers")
    stck = ax2.scatter(year_stock, stock, label = "Stock numbers", color = 'green')
    fig.tight_layout()

    plt.annotate(xy=[2010,250000], s='  MAR introduced')
    plt.plot([2010,2010], [0,700000], color =  'black', linestyle = 'dashed')
    plt.legend([conc, stck], ["Concentration", "Stock numbers"])
    plt.show()

def plot_benchmark():
    ''' Compare analytical and numerical solutions.

        Parameters:
        -----------
        none

        Returns:
        --------
        none

        Notes:
        ------
        This function called within if __name__ == "__main__":

        It should contain commands to obtain analytical and numerical solutions,
        plot these, and either display the plot to the screen or save it to the disk.
    '''
    # Numerical solution
    t, x = solve_ode_concentration(ode_model_concentration, t0 = ?, t1 = ?, dt = ?, x0 =?, pars = [?])

    # Analytical solution
    def y_an(x):
        return (-1+1/(math.exp(x)))

    # set up numerical vectors for independent and dependent 
    y_an_2 = np.vectorize(y_an)
    t_an = np.arange(0, 10, 0.1)

    # plot both graphs on the same axis
    f1, ax = plt.subplots(1,1)
    ax.plot(t, x, 'r', marker = 'x', label = 'Numerical Solution')
    ax.plot(t_an, y_an_2(t_an), 'b:', label = 'Analytical Solution')
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