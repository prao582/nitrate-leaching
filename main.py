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
    ''' Returns time and pressure solved anaylitcally arrays

        Parameters:
        -----------
        t0: Starting year
        t1: Ending year
        dt: Step size (in years)
        b: parameter

        Returns:
        --------
        t :  array-like
                Vector of times (years) at which measurements were taken.
        P : array-like
                Vector of pressure values in Pa
    '''
        
    n = int(np.ceil(t1-t0)/dt)
    t = t0 + np.arange(n+1)*dt
    p = 0.*t

    for i in range (n):
        p[i] = math.exp(-2*b*t[i]-126.4233635)
    return t, p
###################################################
#ODE MODELS
def ode_model_pressure_with_sink(t, P, tmar, Pmar, b, Pa):
    ''' Returns dPdt using the pressure ode provided for the carbon sink scenario

        Parameters:
        -----------
        t: time in years
        P: Pressure in Pa (unknown)
        tmar: when MAR program is implemented (year)
        Pmar: Pressure drop the MAR program produces
        b = recharge coefficient         
        Pa: Pressure drop at high pressure boundary in Pa

        Returns:
        --------
        dPdt: Pressure derivative solved for a point in time (t)
    '''
    
    dPdt = -b * (P + (Pa/2)) - (b * (P - (Pa/2)))
    return dPdt

def ode_model_concentration_with_sink(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar,b):
    ''' Returns dCdt using the pressure ode provided for the carbon sink scenario

        Parameters:
        -----------
        t: time in years
        C: Concentration (unknown)
        n: number of stock 
        P: Pressure in Pa 
        tdelay: time delay in years
        M: Mass Parameter (kg)
        P0: surface pressure parameter in Pa
        a: carbon sink infiltration coefficient parameter 
        b1: infiltration coefficient without carbon sink parameter 
        bc: fresh inflow coefficient
        Pa: Pressure drop at high pressure boundary in Pa
        Pmar: Pressure drop the MAR program produces
        b = recharge coefficient         

        Returns:
        --------
        dCdt: Concentration derivative solved for a point in time (t)
    '''    
    n = stock_interpolation(t-tdelay)
    if (t<2012): # 2012 takes into account the 2 year delay after 2010 when the sink is installed
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M  

def ode_model_concentration_with_mar(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar, b):
    ''' Returns dCdt using the pressure ode provided for the MAR scenario

        Parameters:
        -----------
        t: time in years
        C: Concentration
        n: number of stock 
        P: Pressure in Pa 
        tdelay: time delay in years
        M: Mass Parameter (kg)
        P0: surface pressure parameter in Pa
        a: carbon sink infiltration coefficient parameter 
        b1: infiltration coefficient without carbon sink parameter 
        bc: fresh inflow coefficient
        Pa: Pressure drop at high pressure boundary in Pa
        Pmar: Pressure drop the MAR program produces in Pa
        b = recharge coefficient         

        Returns:
        --------
        dCdt: Concentration derivative solved for a point in time (t)
    '''  
    n = stock_interpolation(t-tdelay)
    n = stock_interpolation(t-tdelay)
    if (t<2012): # 2012 takes into account the 2 year delay after 2010 when the sink is installed
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    elif (t<2020):
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M  
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*(Pa+Pmar)) * C)
        return dCdt /M
    
    return dCdt / M

def ode_model_pressure_with_mar(t, P, tmar, Pmar, b, Pa):
    ''' Returns dPdt using the pressure ode provided for the MAR scenario

        Parameters:
        -----------
        t: time in years
        P: Pressure in Pa (unknown)
        tmar: when MAR program is implemented (year)
        b = recharge coefficient         
        Pa: Pressure drop at high pressure boundary in Pa
        
        Returns:
        --------
        dPdt: Pressure derivative solved for a point in time (t)
    '''
    if (t<2020):
        dPdt = -b * (P + (Pa/2)) - (b * (P - (Pa/2)))
        return dPdt      
    else:
        dPdt = -b * (P + (Pa/2)) - (b * (P - ((Pa+Pmar)/2)))
        return dPdt
 
###################################################
#ODE SOLVERS
def improved_euler_concentration(f, t0, t1, dt, C0, tdelay, tmar, pars):
    ''' Returns array of Concentration and Time solved using Improved Eulers Method

        Parameters:
        -----------
        f: function to solve (concentration ode model)
        t0: Starting year
        t1: Ending year
        dt: Step size (in years)
        C0: Initial concentration
        tdelay: time delay in years
        tmar: when MAR program is implemented (year)
        pars: M, P0, a, b1, bc, Pa, Pmar, b

        Returns:
        --------
        t: Time array in years of step size dt
        c: Concentration array solved for all points in time (t)
    ''' 
	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    c = 0. * t						        	# c array to store concentration
    c[0] = C0							        # Set initial value of Concentration

    if (f == ode_model_concentration_with_sink): # if scenario is with sink installed, call up pressure ode for sink scenario and solve
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])
    else:
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])

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
    ''' Returns array of Pressure and Time solved using Improved Eulers Method

        Parameters:
        -----------
        f: function to solve (pressure ode model)
        t0: Starting year
        t1: Ending year
        dt: Step size (in years)
        p0: surface pressure parameter in Pa
        tmar: when MAR program is implemented (year)
        Pmar: Pressure drop the MAR program produces in Pa
        pars: b, Pa

        Returns:
        --------
        t: Time array in years of step size dt
        p: Pressure array in Pa solved for all points in time (t)    
    ''' 
	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    p = 0.*t						        	# p array to store pressure
    p[0] = p0							        # Set initial value in Pa
    b = pars[0]

	# Iterate over all values of t
    for i in range (steps):
        f0 = f(t[i], p[i], tmar, Pmar, *pars)
        f1 = f(t[i] + dt, p[i] + dt * f0, tmar, Pmar, *pars)
	    # Increment solution by step size x half of each derivative
        p[i+1] = p[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, p

def improved_euler_concentration_no_sink_no_mar(f, t0, t1, dt, C0, tdelay, tmar, pars):
    ''' Returns array of Concentration and Time solved using Improved Eulers Method

        Parameters:
        -----------
        f: function to solve (concentration ode model)
        t0: Starting year
        t1: Ending year
        dt: Step size (in years)
        C0: Initial concentration
        tdelay: time delay in years
        tmar: when MAR program is implemented (year)
        pars: M, P0, a, b1, bc, Pa, Pmar, b

        Returns:
        --------
        t: Time array in years of step size dt
        c: Concentration array solved for all points in time (t)
    ''' 
	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    c = 0. * t						        	# c array to store concentration
    c[0] = C0							        # Set initial value of Concentration

    t, pressure_array = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])
  
    # Iterate over all values of t
    for i in range (steps):
        P = pressure_array[i]
        n = stock_interpolation(t[i]-tdelay)
        f0 = f(t[i], c[i], n, P, tdelay, *pars)
        f1 = f(t[i] + dt, c[i] + dt * f0, n, P, tdelay, *pars)
	    # Increment solution by step size x half of each derivative
        c[i+1] = c[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, c

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
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    plt.plot(t, C)
    plt.show()

def plot_concentration_model_mar():
    t, C = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    plt.plot(t, C)
    plt.show()

def plot_concentration_model_with_curve_fit():  
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()
    
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 0.3, 0.0001, 0.0003, 1e5, 0, -0.03466])
    ci = np.interp(t, year_conc, concentration)
    global cc, cov
    cc,cov = curve_fit(fit_concentration, t, ci)
    print(cc)
    a = cc[0]
    b = cc[1]
    c = cc[2]
    #                                                                                                                                    M, P0, a, b1, bc, Pa, Pmar, b
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020,  pars = [1e9, 50000,a, b,c , 100000, 0, -0.03466])
    plt.plot(t, C)
    plt.show()

def fit_concentration(t,a,b,c):
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 50000, a,b, c, 100000, 0, -0.03466])
    return C

def plot_sink_and_given():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

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

    t, C = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

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

def plot_no_sink_and_given():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')


    t, C = improved_euler_concentration_no_sink_no_mar(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

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

def plot_sink_and_no_sink_and_given():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    fig, ax1 = plt.subplots()
    plt.title("Stock numbers and concentration against time")
    ax1.set_xlabel("time [years]")
    ax1.set_ylabel("Concentration [mg/L]")
    conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')

    # plots with sink scenario
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    plt.plot(t, C, color = 'black')

    # plots no sink scenario
    t, C = improved_euler_concentration_no_sink_no_mar(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    plt.plot(t, C, color = 'red', linestyle = 'dashed')

    

    ax2 = ax1.twinx()
    ax2.set_xlabel("time, [years]")
    ax2.set_ylabel("Stock numbers")
    stck = ax2.scatter(year_stock, stock, label = "Stock numbers", color = 'green')
    fig.tight_layout()

    
    plt.annotate(xy=[2010,250000], s=' Sink introduced')
    plt.plot([2010,2010], [0,700000], color =  'black', linestyle = 'dashed')
    plt.legend([conc, stck], ["Concentration", "Stock numbers"])
    plt.show()

###################################################
#BENCHMARKING
def plot_benchmark_pressure():
    t, P_Numerical = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, p0 = 50000, tmar = 2020, Pmar = 50000, pars = [-0.03466,100000])
    
    P_Analytical = np.zeros(len(P_Numerical))
    P_Error = np.zeros(len(P_Numerical))
    inverse_stepsize = np.linspace(1, 11, 11)
    P_Convergence = np.zeros(len(inverse_stepsize))

    for i in range (len(P_Numerical)):
        P_Analytical[i] = math.exp(-2*-0.03466*t[i]-126.4233635)
        P_Error[i] = abs(P_Analytical[i] - P_Numerical[i])

    for i in range (len(inverse_stepsize)):
        tA, PA = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 1980, t1 = 2019, dt = inverse_stepsize[i]**(-1), p0 = 50000, tmar = 2020, Pmar = 50000, pars = [-0.03466,100000])
        P_Convergence[i] = PA[-1]
           
    # plt.subplot(1,3,2)
    # plt.plot(t,P_Error,'k-')
    # plt.title('Error Analysis')
    # plt.xlabel('t')
    # plt.ylabel('Relative Error Against Benchmark')

    plt.plot(t,P_Numerical,'bD',label = 'Numerical')
    plt.plot(t,P_Analytical,'r+',label = 'Analytical')
    plt.title('Pressure Numerical vs Analytical Benchmark')
    plt.xlabel('Time (Years)')
    plt.ylabel('Pressure (Pa)')
    plt.legend()
    plt.show()
    
    plt.plot(inverse_stepsize,P_Convergence,'bx')
    plt.title('Pressure Timestep Convergence')
    plt.xlabel('1/delta t')
    plt.ylabel('Solution(t=2019)')
    plt.legend()
    plt.show()

def plot_benchmark_concentration():
    # M = 1e9 # mass parameter (estimated)
    # tdelay = 2 #time delay in years parameter (given)
    # P0 = 50000 #surface pressure parameter in Pa (given)
    # Pa = 100000 #pressure drop at high pressure boundary in Pa(given)
    # a = 0.3 #carbon sink infiltration coefficient parameter (justin gave)
    # b1 = 0.0001 #infiltration coefficient without carbon sink parameter (justin gave)
    # bc = 0.0003 #fresh inflow coefficient (to calibrate)
    #Pmar = 0 #pressure due to mar operation (0 for only sink implementation)
    # b = 0.003 #recharge coefficient (to calibrate)
    
    M = 1e9
    P0 = 5e4
    a = 6.50424868e-01
    b1 = 7.35181289e-01
    bc = -3.39986410e+04
    Pa = 1e5
    Pmar = 0
    b = -0.03466
    
    # Numerical solution                                                                                                                        t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar,b
    t, C_Numerical = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    #t, C_Numerical = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1, 0, 1 , 1, -1, 0, 0, -1])

    C_Analytical = np.zeros(len(C_Numerical))
    C_Error = np.zeros(len(C_Numerical))
    inverse_stepsize = np.linspace(1, 3, 31)
    C_Convergence = np.zeros(len(inverse_stepsize))
    
    k = bc
    c = -126.4233635
    j = 2*b
    c1 = 0
    h = 2
    
    
    #finding error between analytical and numerical solutions
    for i in range (len(C_Numerical)):
        if t[i]<2012:
            g = stock_interpolation(t[i]) *b1
        else:
            g = stock_interpolation(t[i]) *b1 * a
        C_Analytical[i] = (math.exp((-k*math.exp(c-j*t[i]) + k*j)/(j*M)/k)) - (g/k)
        #C_Analytical[i] = math.exp(-math.exp(-t[i])) -g 
        C_Error[i] = abs(C_Analytical[i] - C_Numerical[i])

    #convergence
    for i in range (0, len(inverse_stepsize)):
        tA, CA = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = inverse_stepsize[i]**(-1), C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
        #tA, CA = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = inverse_stepsize[i]**(-1), C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1, 0, 0.5 , 1, -1, 0, 0, -0.03466])
        C_Convergence[i] = CA[-1]
  
    plt.plot(t,C_Numerical,'b--',label = 'Numerical')
    plt.plot(t,C_Analytical,'rx',label = 'Analytical')
    plt.title('Concentration Numerical vs Analytical Benchmark')
    plt.xlabel('Time (Years)')
    plt.ylabel('Concentraion (mg/L)')
    plt.legend()
    plt.show()
    
    plt.plot(inverse_stepsize,C_Convergence,'bx')
    plt.title('Concentration Timestep Convergence')
    plt.xlabel('1/delta t')
    plt.ylabel('Solution(t=2019)')
    plt.legend()
    plt.show()

###################################################
#forecasting
def extrapolate_stock_growth(t):

    year, stock = stock_population()
    
    years_copy = year.copy()
    years_copy = np.append(years_copy,2030.5)

    stock1 = stock.copy()
    stock1 = np.append(stock,2e6)

    n1 = np.interp(t, years_copy, stock1)
    
    return n1

def extrapolate_stock_maintain(t):

    year, stock = stock_population()
    
    years_copy = year.copy()
    years_copy = np.append(years_copy,2030.5)

    stock1 = stock.copy()
    stock1 = np.append(stock,640000)

    n1 = np.interp(t, years_copy, stock1)
    
    return n1

def ode_model_concentration_with_sink_stock_growth(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar,b):
    ''' Returns dCdt using the pressure ode provided for the carbon sink scenario

        Parameters:
        -----------
        t: time in years
        C: Concentration (unknown)
        n: number of stock 
        P: Pressure in Pa 
        tdelay: time delay in years
        M: Mass Parameter (kg)
        P0: surface pressure parameter in Pa
        a: carbon sink infiltration coefficient parameter 
        b1: infiltration coefficient without carbon sink parameter 
        bc: fresh inflow coefficient
        Pa: Pressure drop at high pressure boundary in Pa
        Pmar: Pressure drop the MAR program produces
        b = recharge coefficient         

        Returns:
        --------
        dCdt: Concentration derivative solved for a point in time (t)
    '''     
    n = extrapolate_stock_growth(t-tdelay)
    if (t<2012): # 2012 takes into account the 2 year delay after 2010 when the sink is installed
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M  

def ode_model_concentration_with_sink_stock_maintain(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar,b):
    ''' Returns dCdt using the pressure ode provided for the carbon sink scenario

        Parameters:
        -----------
        t: time in years
        C: Concentration (unknown)
        n: number of stock 
        P: Pressure in Pa 
        tdelay: time delay in years
        M: Mass Parameter (kg)
        P0: surface pressure parameter in Pa
        a: carbon sink infiltration coefficient parameter 
        b1: infiltration coefficient without carbon sink parameter 
        bc: fresh inflow coefficient
        Pa: Pressure drop at high pressure boundary in Pa
        Pmar: Pressure drop the MAR program produces
        b = recharge coefficient         

        Returns:
        --------
        dCdt: Concentration derivative solved for a point in time (t)
    '''     
    n = extrapolate_stock_maintain(t-tdelay)
    if (t<2012): # 2012 takes into account the 2 year delay after 2010 when the sink is installed
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M

def ode_model_concentration_with_mar_stock_growth(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar, b):
    ''' Returns dCdt using the pressure ode provided for the MAR scenario

        Parameters:
        -----------
        t: time in years
        C: Concentration
        n: number of stock 
        P: Pressure in Pa 
        tdelay: time delay in years
        M: Mass Parameter (kg)
        P0: surface pressure parameter in Pa
        a: carbon sink infiltration coefficient parameter 
        b1: infiltration coefficient without carbon sink parameter 
        bc: fresh inflow coefficient
        Pa: Pressure drop at high pressure boundary in Pa
        Pmar: Pressure drop the MAR program produces in Pa
        b = recharge coefficient         

        Returns:
        --------
        dCdt: Concentration derivative solved for a point in time (t)
    '''  
    n = extrapolate_stock_growth(t-tdelay) 
    if (t<2012): # 2012 takes into account the 2 year delay after 2010 when the sink is installed
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    elif (t<2020):
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*(Pa+Pmar)) * C)
        return dCdt / M


def ode_model_concentration_with_mar_stock_maintain(t, C, n, P, tdelay, M, P0, a, b1, bc, Pa, Pmar, b):
    ''' Returns dCdt using the pressure ode provided for the MAR scenario

        Parameters:
        -----------
        t: time in years
        C: Concentration
        n: number of stock 
        P: Pressure in Pa 
        tdelay: time delay in years
        M: Mass Parameter (kg)
        P0: surface pressure parameter in Pa
        a: carbon sink infiltration coefficient parameter 
        b1: infiltration coefficient without carbon sink parameter 
        bc: fresh inflow coefficient
        Pa: Pressure drop at high pressure boundary in Pa
        Pmar: Pressure drop the MAR program produces in Pa
        b = recharge coefficient         

        Returns:
        --------
        dCdt: Concentration derivative solved for a point in time (t)
    '''  
    n = extrapolate_stock_maintain(t-tdelay)
    if (t<2012): # 2012 takes into account the 2 year delay after 2010 when the sink is installed
        dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M        
    elif (t<2020):
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
        return dCdt / M
    else:
        dCdt = (n * a * b1 * (P-P0)) + (bc * (P - 0.5*(Pa+Pmar)) * C)
        return dCdt / M

def improved_euler_concentration_growth(f, t0, t1, dt, C0, tdelay, tmar, pars):
    ''' Returns array of Concentration and Time solved using Improved Eulers Method

        Parameters:
        -----------
        f: function to solve (concentration ode model)
        t0: Starting year
        t1: Ending year
        dt: Step size (in years)
        C0: Initial concentration
        tdelay: time delay in years
        tmar: when MAR program is implemented (year)
        pars: M, P0, a, b1, bc, Pa, Pmar, b

        Returns:
        --------
        t: Time array in years of step size dt
        c: Concentration array solved for all points in time (t)
    ''' 
	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    c = 0. * t						        	# c array to store concentration
    c[0] = C0							        # Set initial value of Concentration

    if (f == ode_model_concentration_with_sink_stock_growth): # if scenario is with sink installed, call up pressure ode for sink scenario and solve
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 2019, t1 = 2030, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])
    else:
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_mar, t0 = 2019, t1 = 2030, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])

    # Iterate over all values of t
    for i in range (steps):
        P = pressure_array[i]
        n = extrapolate_stock_growth(t[i]-tdelay)
        f0 = f(t[i], c[i], n, P, tdelay, *pars)
        f1 = f(t[i] + dt, c[i] + dt * f0, n, P, tdelay, *pars)
	    # Increment solution by step size x half of each derivative
        c[i+1] = c[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, c

def improved_euler_concentration_maintain(f, t0, t1, dt, C0, tdelay, tmar, pars):
    ''' Returns array of Concentration and Time solved using Improved Eulers Method

        Parameters:
        -----------
        f: function to solve (concentration ode model)
        t0: Starting year
        t1: Ending year
        dt: Step size (in years)
        C0: Initial concentration
        tdelay: time delay in years
        tmar: when MAR program is implemented (year)
        pars: M, P0, a, b1, bc, Pa, Pmar, b

        Returns:
        --------
        t: Time array in years of step size dt
        c: Concentration array solved for all points in time (t)
    ''' 
	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    c = 0. * t						        	# c array to store concentration
    c[0] = C0							        # Set initial value of Concentration

    if (f == ode_model_concentration_with_sink_stock_maintain): # if scenario is with sink installed, call up pressure ode for sink scenario and solve
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_sink, t0 = 2019, t1 = 2030, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])
    else:
        t, pressure_array = improved_euler_pressure(ode_model_pressure_with_mar, t0 = 2019, t1 = 2030, dt = 0.1, p0 = 50000, tmar = tmar, Pmar = pars[6], pars = [-0.03466,100000])

    # Iterate over all values of t
    for i in range (steps):
        P = pressure_array[i]
        n = extrapolate_stock_maintain(t[i]-tdelay)
        f0 = f(t[i], c[i], n, P, tdelay, *pars)
        f1 = f(t[i] + dt, c[i] + dt * f0, n, P, tdelay, *pars)
	    # Increment solution by step size x half of each derivative
        c[i+1] = c[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, c

def plot_forecasting():
    #give data and concentration models
    t_conc,conc = nitrate_concentration()
    t_conc_model_sink, conc_sink = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    t_conc_model_mar, conc_mar = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

    figure,ax = plt.subplots(1,1)
    ax.plot(t_conc, conc, 'r',marker = '.',linestyle = 'None', label = 'Concentration data')
    ax.plot(t_conc_model_sink, conc_sink, 'g', label = 'Conc model sink')
    ax.plot(t_conc_model_mar, conc_mar, 'b', label = 'Conc model mar')
    
    #what-if scenarios
    
    #Rejection of consent, no mar and maintain stock number
    ts1,cs1 = improved_euler_concentration_maintain(ode_model_concentration_with_sink_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    ax.plot(ts1, cs1, 'm', label = 'Scenario One, Rejection of consent, so carbon sink and maintaining stock number')

    #Rejection of consent, no mar and increase stock number
    ts2,cs2 = improved_euler_concentration_growth(ode_model_concentration_with_sink_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    ax.plot(ts2, cs2, 'k', label = 'Scenario Two, Rejection of consent, so carbon sink and and increase stock number')

    #Acceptance of consent, so implement mar and maintain stock number
    ts3,cs3 = improved_euler_concentration_maintain(ode_model_concentration_with_mar_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, -5000, -0.03466])
    ax.plot(ts3, cs3, 'y', label = 'Scenario Three, Acceptance of consent, so implement mar and maintain stock number', linestyle = 'dashed')

    #Acceptance of consent, so implement mar and increase stock number
    ts4,cs4 = improved_euler_concentration_growth(ode_model_concentration_with_mar_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, -5000, -0.03466])
    ax.plot(ts4, cs4, 'c', label = 'Scenario Four, Acceptance of consent, so implement mar and increase stock number', linestyle = 'dashed')

    plt.plot([1980,2030], [13,13], label = 'Maximum Allowable Nitrate', color =  'black', linestyle = 'dashed')
    ax.legend()
    ax.set(title = 'Concentration forecasting', xlabel = 'Time (years)', ylabel = 'Concentration')
    plt.show()

###################################################
#UNCERTAINITY
def uncertainity():
    lines = 5
    var = 10
    global cc, cov
 #[ 6.50424612e-01  7.42921705e-01 -3.43565993e+04]
    a = 6.50424612e-01
    b = 7.42921705e-01
    c = -3.43565993e+04
    a_var = a/var
    b_var = b/var
    f,ax = plt.subplots(1,1)

    #generate the nomral distribution for parameters
    a_norm = np.random.normal(a, a_var, lines)
    b_norm = np.random.normal(b, b_var, lines)
    #c_norm = np.random.normal(c, c_var, lines)

    #give data and concentration models
    t_conc,conc = nitrate_concentration()
    t_conc_model_sink, conc_sink = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    t_conc_model_mar, conc_mar = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, -5000, -0.03466])

    figure,ax = plt.subplots(1,1)
    ax.plot(t_conc, conc, 'r',marker = '.',linestyle = 'None', label = 'Concentration data')
    ax.plot(t_conc_model_sink, conc_sink, 'b', label = 'Conc model sink')
   # ax.plot(t_conc_model_mar, conc_mar, 'b', label = 'Conc model mar')
    
    #what-if scenario 1
    for i in range(lines):
        pars_C = [a_norm[i], b_norm[i]]
        ts1,cs1 = improved_euler_concentration_maintain(ode_model_concentration_with_sink_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, *pars_C, -3.39986410e+04, 1e5, 0, -0.03466])
        plt.plot(ts1,cs1, alpha = 0.2, color = 'coral')

    ts1,cs1 = improved_euler_concentration_maintain(ode_model_concentration_with_sink_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    ax.plot(ts1, cs1, color = 'coral', label = 'Scenario One, rejection of consent, maintaining stock number', linewidth = 2)

    #what-if scenario 2
    for i in range(lines):
        pars_C = [a_norm[i], b_norm[i]]
        ts2,cs2 = improved_euler_concentration_growth(ode_model_concentration_with_sink_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, *pars_C, -3.39986410e+04, 1e5, 0, -0.03466])
        plt.plot(ts2,cs2, color = 'cyan',alpha = 0.2)

    ts2,cs2 = improved_euler_concentration_growth(ode_model_concentration_with_sink_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    ax.plot(ts2, cs2, 'cyan', label = 'Scenario Two, rejection of consent, double stock number', linewidth =2)

    #what-if scenario 3
    for i in range(lines):
        pars_C = [a_norm[i], b_norm[i]]
        ts3,cs3 = improved_euler_concentration_maintain(ode_model_concentration_with_mar_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, *pars_C, -3.39986410e+04, 1e5, -5000, -0.03466])
        plt.plot(ts3,cs3, color = 'purple', alpha = 0.2)

    ts3,cs3 = improved_euler_concentration_maintain(ode_model_concentration_with_mar_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, -5000, -0.03466])
    ax.plot(ts3, cs3, 'purple', label = 'Scenario Three, acceptance of consent, implement mar and maintain stock number', linewidth = 2)

    #what-if scenario 4
    for i in range(lines):
        pars_C = [a_norm[i], b_norm[i]]
        ts4,cs4 = improved_euler_concentration_growth(ode_model_concentration_with_mar_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, *pars_C, -3.39986410e+04, 1e5, -5000, -0.03466])
        plt.plot(ts4,cs4, color = 'green',alpha = 0.2)

    ts4,cs4 = improved_euler_concentration_growth(ode_model_concentration_with_mar_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, -5000, -0.03466])
    ax.plot(ts4, cs4, 'green', label = 'Scenario Four, acceptance of consent, implement mar and double stock number', linewidth = 2)
    ax.legend()

    plt.show()

def uncertainty_pranav():
    mass = 1e9
    P0 = 5e4
    a = 6.50424868e-01
    b1 = 7.35181289e-01
    bc = 3.39986410e+04
    Pa = 1e5
    Pmar = 0
    Pmar1 = -5000
    b = 0.03466
    
    # Creating arrays to store max values
    # Max lines = 1000
    lines = 200
    C1_max = np.zeros(lines)
    C2_max = np.zeros(lines)
    C3_max = np.zeros(lines)
    C4_max = np.zeros(lines)

    # Calculating variance for each parameter
    var = 10
    a_var = a/var
    b1_var = b1/var
    bc_var = bc/var
    b_var = b/var

    # Generating normal distributions for parameters
    a_norm = np.random.normal(a, a_var, lines)
    b1_norm = np.random.normal(b1, b1_var, lines)
    bc_norm = np.random.normal(bc, bc_var, lines)
    b_norm = np.random.normal(b, b_var, lines)
    
    
    figure,ax = plt.subplots(1,1)
    for i in range(lines):
        pars_C1 = [mass, P0, a_norm[i], b1_norm[i], -bc_norm[i], Pa, Pmar, -b_norm[i]]
        pars_C2 = [mass, P0, a_norm[i], b1_norm[i], -bc_norm[i], Pa, Pmar1, -b_norm[i]]
        #give data and concentration models
        t_conc,conc = nitrate_concentration()
        t_conc_model_sink, conc_sink = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
        t_conc_model_mar, conc_mar = improved_euler_concentration(ode_model_concentration_with_mar, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])

        
        ax.plot(t_conc, conc, 'r',marker = '.',linestyle = 'None', label = 'Concentration data')
        ax.plot(t_conc_model_sink, conc_sink, 'g', label = 'Conc model sink')
        ax.plot(t_conc_model_mar, conc_mar, 'b', label = 'Conc model mar')
    
        #what-if scenarios
    
        #Rejection of consent, no mar and maintain stock number
        ts1,cs1 = improved_euler_concentration_maintain(ode_model_concentration_with_sink_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = pars_C1)
        ax.plot(ts1, cs1, 'm', label = 'Scenario One, Rejection of consent, so carbon sink and maintaining stock number')

        #Rejection of consent, no mar and increase stock number
        ts2,cs2 = improved_euler_concentration_growth(ode_model_concentration_with_sink_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_sink[-1], tdelay = 2, tmar = 2020, pars = pars_C1)
        ax.plot(ts2, cs2, 'k', label = 'Scenario Two, Rejection of consent, so carbon sink and and increase stock number')

        #Acceptance of consent, so implement mar and maintain stock number
        ts3,cs3 = improved_euler_concentration_maintain(ode_model_concentration_with_mar_stock_maintain, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = pars_C2)
        ax.plot(ts3, cs3, 'y', label = 'Scenario Three, Acceptance of consent, so implement mar and maintain stock number', linestyle = 'dashed')

        #Acceptance of consent, so implement mar and increase stock number
        ts4,cs4 = improved_euler_concentration_growth(ode_model_concentration_with_mar_stock_growth, t0 = 2019, t1 = 2030, dt = 0.1, C0 = conc_mar[-1], tdelay = 2, tmar = 2020, pars = pars_C2)
        ax.plot(ts4, cs4, 'c', label = 'Scenario Four, Acceptance of consent, so implement mar and increase stock number', linestyle = 'dashed')

        plt.plot([1980,2030], [13,13], label = 'Maximum Allowable Nitrate', color =  'black', linestyle = 'dashed')
    
    ax.legend()
    ax.set(title = 'Concentration forecasting', xlabel = 'Time (years)', ylabel = 'Concentration')
    plt.show()




def plot_concentration_misfit():
    year_conc = np.genfromtxt("nl_n.csv", delimiter = ',', skip_header = 1, usecols = 0)
    concentration = np.genfromtxt('nl_n.csv', delimiter = ',', skip_header = 1, usecols = 1)
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, pars = [1e9, 5e4, 6.50424868e-01 , 7.35181289e-01, -3.39986410e+04, 1e5, 0, -0.03466])
    concentration_misfit = concentration - np.interp(year_conc, t, C)
    plt.plot(t, concentration_misfit, 'rx')
    plt.plot(t, np.zeros(len(t)), 'k.')
    plt.set_title("Concentration Misfit")
    plt.set_ylabel("conc (mg /L)")
    plt.set_xlabel('time (t)')
    plt.show()
    
if __name__ == "__main__":
    #ode_model_pressure()
    #ode_model_concentration()
    #stock_population()
    # plot_given_data()
    #plot_concentration_model_sink()
    #plot_concentration_model_mar()
    #plot_sink_and_given()
    #plot_mar_and_given()
    #plot_concentration_model_with_curve_fit()
    #n = stock_interpolation(2000.96)
    #print(n)
    #plot_benchmark_concentration()
    #plot_benchmark_pressure()
    #plot_forecasting()
    #uncertainty_pranav()
    #uncertainity()
    #plot_pressure_model_sink()
    #plot_pressure_model_mar()
    #plot_sink_and_no_sink_and_given()
    plot_concentration_misfit()