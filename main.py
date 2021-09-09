# imports
from typing import TypeVar
import numpy as np
from matplotlib import pyplot as plt
import math as math
from numpy.core.arrayprint import dtype_is_implied
from numpy.testing._private.utils import break_cycles
import scipy as scipy
from scipy.optimize import curve_fit
###################################################
#read in functions
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
###################################################
#interpolation functions
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
def concentration_interpolation(t):
    ''' Return nitrate concentration parameter n for model

        Parameters:
        -----------
        t : array-like
            Vector of times at which to interpolate the nitrate concentration

        Returns:
        --------
        n : array-like
            Nitrate concentration interpolated at t.
    '''

    year, concentration = nitrate_concentration()

    n = np.interp(t, year, concentration)

    return n    
###################################################
#analytically solves the pressure ode to find P
def pressure_analytical(t0,t1,dt,b):
    n = int(np.ceil(t1-t0)/dt)
    t = t0 + np.arange(n+1)*dt
    p = 0.*t

    for i in range (n):
        p[i] = math.exp(-2*b*t[i]-126.4233635)
    return t, p
###################################################
#ODE MODELS
def ode_model_pressure_with_sink(t, P, tmar, Pmar, b, Pa):

    dPdt = -b * (P + (Pa/2)) - (b * (P - (Pa/2)))
    return dPdt

def ode_model_concentration_with_sink(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar,b):
    #parameters: M, tdelay, P0, bc, a, b1, Pa, Pmar
    #inputs: C, t
    #called inputs: n
    n = stock_interpolation(t-tdelay)
    if (t<2012):
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M  

def ode_model_concentration_with_mar(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar, b):
    #parameters: M, tdelay, P0, bc, a, b1, Pa, Pmar
    #inputs: C, t
    #called inputs: n
    n = stock_interpolation(t-tdelay)
    dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
    
    return dCdt / M

def ode_model_pressure_with_mar(t, P, tmar, Pmar, b, Pa):
    #parameters: M, tdelay, P0, bc, a, b1, Pa, Pmar
    #inputs: C, t
    #called inputs: n
    if (t<tmar):
        dPdt = -b * (P + (Pa/2)) - (b * (P - (Pa/2)))
        return dPdt      
    else:
        dPdt = -b * (P + (Pa/2)) - (b * (P - ((Pa+Pmar)/2)))
        return dPdt
 
###################################################
#ODE SOLVERS
def improved_euler_concentration(f, t0, t1, dt, C0, tdelay, tmar, Pmar, pars):

	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    c = 0. * t						        	# c array to store concentration
    c[0] = C0							        # Set initial value

    if (f == ode_model_concentration_with_sink):
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = Pmar,  pars = [-0.03466,100000])
    else:
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = Pmar, pars = [-0.03466,100000])

    # Iterate over all values of t
    for i in range (steps):
        P = pressure_array[i]
        n = stock_interpolation(t[i]-tdelay)
        f0 = f(t[i], c[i], n, P, tdelay, *pars)
        f1 = f(t[i] + dt, c[i] + dt * f0, n, P, tdelay, *pars)
	    # Increment solution by step size x half of each derivative
        c[i+1] = c[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, c

def improved_euler_pressure(f, t0, t1, dt, p0, tmar, Pmar, pars):
	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    p = 0.*t						        	# p array to store pressure
    p[0] = 50000							    # Set initial value in Pa
    b = pars[0]
    t, p_numerical = pressure_analytical(t0,t1,dt,b) # solves the pressure numerically

	# Iterate over all values of t
    for i in range (steps):
        f0 = f(t[i], p_numerical[i], tmar, Pmar, *pars)
        f1 = f(t[i] + dt, p_numerical[i] + dt * f0, tmar, Pmar, *pars)
	    # Increment solution by step size x half of each derivative
        p[i+1] = p[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, p
###################################################
#PLOTTING
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

    plt.annotate(xy=[2010,250000], s=' Sink introduced')
    plt.plot([2010,2010], [0,700000], color =  'black', linestyle = 'dashed')
    plt.legend([conc, stck], ["Concentration", "Stock numbers"])
    plt.show()

def plot_pressure_model_sink():
    t, P = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = 2020, Pmar = 50000, pars = [-0.03466,100000])
    plt.plot(t, P)
    plt.title('Pressure')
    plt.xlabel('t')
    plt.ylabel('Pressure (Pa)')
    plt.show()

def plot_pressure_model_mar():
    t, P = improved_euler_pressure(ode_model_pressure_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar =2020, Pmar = 50000, pars = [-0.03466,100000])
    plt.plot(t, P)
    plt.title('Pressure')
    plt.xlabel('t')
    plt.ylabel('Pressure (Pa)')
    plt.show()

def plot_concentration_model_sink():
    #[ 4.54391383e-01  5.30450263e-02 -1.37058256e+03]
    #[ 6.49474256e-01  8.70926035e-01 -3.12050719e+04]
    #[ 6.50424868e-01  7.35181289e-01 -3.39986410e+04]
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    plt.plot(t, C)
    plt.show()

def plot_concentration_model_mar():
    t, C = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    plt.plot(t, C)
    plt.show()

def plot_concentration_model_with_curve_fit():  
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()
    
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 5e4, 0.3, 0.0001, 0.0003, 1e5, 0, -0.03466])
    ci = np.interp(t, year_conc, concentration)
    
    cc,_ = curve_fit(fit_concentration, t, ci)
    print(cc)
    a = cc[0]
    b = cc[1]
    c = cc[2]
    #                                                                                                                                    M, P0, a, b1, bc, Pa, Pmar, b
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 50000,a, b,c , 100000, 0, -0.03466])
    plt.plot(t, C)
    plt.show()
    #   [ 4.54391383e-01  5.30450263e-02 -1.37058256e+03]
    #-0.18463230481350532
    #0.01674194787021388

def fit_concentration(t,a,b,c):
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 50000, a,b, c, 100000, 0, -0.03466])
    return C

def plot_sink_and_given():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')
    #t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, pars = [1e9, 50000, -0.18463230481350532,0.01674194787021388 ,30, 100000, 0, 0.003])
    #t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, pars = [1e9, 5e4, 0.45, 0.053, -1370, 1e5, 0, 0.003])
    #t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, pars = [1e10, 5e4, 0.65, 0.87, -40000, 1e5, 0, 0.03466])
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

    plt.plot(t, C, color = 'black')

    ax2 = ax1.twinx()
    ax2.set_xlabel("time, [years]")
    ax2.set_ylabel("Stock numbers")
    stck = ax2.scatter(year_stock, stock, label = "Stock numbers", color = 'green')
    fig.tight_layout()

    
    plt.annotate(xy=[2010,250000], s=' Sink introduced')
    plt.plot([2010,2010], [0,700000], color =  'black', linestyle = 'dashed')
    plt.legend([conc, stck], ["Concentration", "Stock numbers"])
    plt.show()

def plot_mar_and_given():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')

    t, C = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

    plt.plot(t, C, color = 'black')

    ax2 = ax1.twinx()
    ax2.set_xlabel("time, [years]")
    ax2.set_ylabel("Stock numbers")
    stck = ax2.scatter(year_stock, stock, label = "Stock numbers", color = 'green')
    fig.tight_layout()

    
    plt.annotate(xy=[2020,250000], s=' MAR introduced')
    plt.plot([2020,2020], [0,700000], color =  'black', linestyle = 'dashed')
    plt.legend([conc, stck], ["Concentration", "Stock numbers"])
    plt.show()

###################################################
#BENCHMARKING
def plot_benchmark():
    # M = 1e9 # mass parameter (estimated)
    # tdelay = 2 #time delay in years parameter (given)
    # P0 = 50000 #surface pressure parameter in Pa (given)
    # Pa = 100000 #pressure drop at high pressure boundary in Pa(given)
    # a = 0.3 #carbon sink infiltration coefficient parameter (justin gave)
    # b1 = 0.0001 #infiltration coefficient without carbon sink parameter (justin gave)
    # bc = 0.0003 #fresh inflow coefficient (to calibrate)
    Pmar = 0 #pressure due to mar operation (0 for only sink implementation)
    # b = 0.003 #recharge coefficient (to calibrate)
    M = 1e9
    P0 = 5e4
    a = 6.50424868e-01
    b1 = 7.35181289e-01
    bc = -3.39986410e+04
    Pa = 1e5
    b = -0.03466
    # Numerical solution
    t, C_Numerical = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

    C_Analytical = np.zeros(len(C_Numerical))
    C_Error = np.zeros(len(C_Numerical))
    inverse_stepsize = np.linspace(1, 3, 21)
    C_Convergence = np.zeros(len(inverse_stepsize))
    # Analytical solution
    #def cu_an(x):
    #    return (math.exp(-x))

    #cu_vector = np.vectorize(cu_an)
    #C_Analytical = cu_vector(t)

    #REMEMBER TO ADD A BOOLEAN OPERATOR FOR WHEN THE ODE CHANGES TO THE SINK VERSION
    k = bc
    c = -126.4233635
    j = 2*b
    #c1
    
    
    #finding error between analytical and numerical solutions
    for i in range (len(C_Numerical)):
        if t[i]<2012:
            g = stock_interpolation(t[i])*b1
        else:
            g = stock_interpolation(t[i])*b1 * a
        C_Analytical[i] = (math.exp((-k*math.exp(c-j*t[i]) + k*j)/(j*M)/k)) - (g/k)
        C_Error[i] = abs(C_Analytical[i] - C_Numerical[i])

    #convergence
    for i in range (0, len(inverse_stepsize)):
        tA, CA = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = inverse_stepsize[i]**(-1), C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 0, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
        C_Convergence[i] = CA[-1]
        
        



    plt.subplot(1,3,1)
    plt.plot(t,C_Numerical,'b--',label = 'Numerical')
    plt.plot(t,C_Analytical,'rx',label = 'Analytical')
    plt.legend()
    plt.title('Benchmark')
    plt.xlabel('t')
    plt.ylabel('C')
    #plt.show()
    plt.subplot(1,3,2)
    plt.plot(t,C_Error,'k-')
    plt.title('Error Analysis')
    plt.xlabel('t')
    plt.ylabel('Relative Error Against Benchmark')

    plt.subplot(1,3,3)
    plt.plot(inverse_stepsize,C_Convergence,'bx')
    plt.title('Timestep Convergence')
    plt.xlabel('1/delta t')
    plt.ylabel('')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #ode_model_pressure()
    #ode_model_concentration()
    #stock_population()
    #plot_given_data()
    #plot_pressure_model()
    #plot_concentration_model()
    #plot_sink_and_given()
    #plot_mar_and_given()
    #plot_concentration_model_with_curve_fit()
    #n = stock_interpolation(2000.96)
    #print(n)
    plot_benchmark()
