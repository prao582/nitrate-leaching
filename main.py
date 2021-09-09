# imports
from typing import TypeVar
import numpy as np
from matplotlib import pyplot as plt
import math as math
from numpy.core.arrayprint import dtype_is_implied
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
    ''' Returns time and pressure solved numerically arrays

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
    dCdt = (n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
    
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
    if (t<tmar):
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
    
    cc,_ = curve_fit(fit_concentration, t, ci)
    print(cc)
    a = cc[0]
    b = cc[1]
    c = cc[2]
    #                                                                                                                                    M, P0, a, b1, bc, Pa, Pmar, b
    t, C = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, tdelay = 2, tmar = 2020, Pmar = 50000, pars = [1e9, 50000,a, b,c , 100000, 0, -0.03466])
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

###################################################
#BENCHMARKING
def plot_benchmark():
    M = 1e9 # mass parameter (estimated)
    tdelay = 2 #time delay in years parameter (given)
    P0 = 50000 #surface pressure parameter in Pa (given#pressure drop at high pressure boundary in Pa(given))
    Pa = 100000 
    a = 0.3 #carbon sink infiltration coefficient parameter (justin gave)
    b1 = 0.0001 #infiltration coefficient without carbon sink parameter (justin gave)
    bc = 0.0003 #fresh inflow coefficient (to calibrate)
    Pmar = 0 #pressure due to mar operation (0 for only sink implementation)
    b = 0.003 #recharge coefficient (to calibrate)


    # Numerical solution
    t, C_Numerical = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2019, dt = 0.1, C0 = 0.2, pars = [M, P0, a, b1, bc, Pa, Pmar, b])

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

    
    #finding error between analytical and numerical solutions
    for i in range (len(C_Numerical)):
        C_Analytical[i] = math.exp()
        C_Error[i] = abs(C_Analytical[i] - C_Numerical[i])

    #convergence
    for i in range (len(inverse_stepsize)):
        tA, CA = improved_euler_concentration(ode_model_concentration_with_sink, t0 = 1980, t1 = 2018, dt = inverse_stepsize[i]**(-1), C0 = 0.2, pars = [M, P0, a, b1, bc, Pa, Pmar, b])
        C_Convergence[i] = CA[-1]
        

    #for i in range (len(inverse_stepsize)):
        #tA, CA = improved_euler_concentration(ode_model_concentration, t0 = 1980, t1 = 2018, inverse_stepsize[i]**(-1), C0 = ?, pars = [M, t, tdelay, P, P0, a, b1, bc, C, Pa, Pmar, b])
        #C_Convergence[i] = CA[-1]
        #C_Analytical[i] = 
        #C_Error[i] = abs(C_Analytical[i] - C_Numerical[i])



    plt.subplot(1,3,1)
    plt.plot(t,C_Numerical,'b--',label = 'Numerical')
    plt.plot(t,C_Analytical,'rx',label = 'Analytical')
    plt.legend()
    plt.title('Benchmark')
    plt.xlabel('t')
    plt.ylabel('C')


    plt.show()



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
    plot_sink_and_given()
    plot_mar_and_given()
    #plot_concentration_model_with_curve_fit()
