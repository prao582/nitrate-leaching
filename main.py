# imports
from typing import TypeVar
import numpy as np
from matplotlib import pyplot as plt
import math as math
###################################################
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
###################################################

#ODE MODELS
def ode_model_pressure_no_mar(t, P, b, Pa):
    
    dPdt = -b * (P + (Pa/2)) - (b * (P - (Pa/2)))
    return dPdt
def ode_model_pressure_mar(t, P, b, Pa, Pmar):
    
    dPdt = -b * (P + (Pa/2)) - (b * (P - ((Pa+Pmar)/2)))
    return dPdt

def ode_model_concentration_no_sink_no_mar(t, C, n, M, P, P0, a, b1, bc, Pa, Pmar, b):
    #parameters: M, tdelay, P0, bc, a, b1, Pa, Pmar
    #inputs: C, t
    #called inputs: n
    P = ode_model_pressure_no_mar(P, b, Pa, Pmar)

    dCdt = (-n * b1 * (P-P0)) + (bc * (P - 0.5*Pa) * C)
    
    return dCdt / M
def ode_model_concentration_sink_mar(t, C, n, M, P, P0, a, b1, bc, Pa, Pmar, b):
    #parameters: M, tdelay, P0, bc, a, b1, Pa, Pmar
    #inputs: C, t
    #called inputs: n

    P = ode_model_pressure_mar(P, b, Pa, Pmar)
    dCdt = (-n * a * b * (P-P0)) + (bc * (P - 0.5*(Pa+Pmar)) * C)
    
    return dCdt / M
def ode_model_concentration_sink_no_mar(t, C, n, M, P, P0, a, b1, bc, Pa, Pmar, b):
    #parameters: M, tdelay, P0, bc, a, b1, Pa, Pmar
    #inputs: C, t
    #called inputs: n, P

    P = ode_model_pressure_no_mar(P, b, Pa, Pmar)
    
    dCdt = (-n * a * b * (P-P0)) + (bc * (P - 0.5*Pa) * C)
    
    return dCdt / M



def improved_euler_concentration(f, t0, t1, dt, C0, tdelay, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of concentration.
        t1 : float
            Final time of concentration.
        dt : float
            Time step length.
        C0 : float
            Initial value of concentration.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable concentration vector.
        c : array-like
            Dependent variable concentration vector.
    '''

	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    c = 0. * t						        	# c array to store concentration
    c[0] = C0							        # Set initial value
	
	# Iterate over all values of t
    for i in range (steps):
        if (t[i]<2010):
            f = ode_model_concentration_no_sink_no_mar
        elif (t[i]<2020):
            f = ode_model_concentration_sink_no_mar
        else:
            f = ode_model_concentration_sink_mar
        n = stock_interpolation(t[i]-tdelay)
        f0 = f(t[i], c[i], n, *pars)
        f1 = f(t[i] + dt, c[i] + dt * f0, n, *pars)
	    # Increment solution by step size x half of each derivative
        c[i+1] = c[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, c

def improved_euler_pressure(f, t0, t1, dt, p0, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of pressure.
        t1 : float
            Final time of pressure.
        dt : float
            Time step length.
        P0 : float
            Initial value of pressure.
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        t : array-like
            Independent variable time vector.
        p : array-like
            Dependent variable pressure vector.
    '''

	# initialise
    steps = int(np.ceil((t1-t0) / dt))	       	# Number of Euler steps to take
    t = t0 + np.arange(steps+1) * dt			# t array
    p = 0. * t						        	# p array to store concentration
    p[0] = p0							        # Set initial value

	
	# Iterate over all values of t
    for i in range (steps):
        #if t[i] < 2019:
        #    f = ode_model_pressure_1
        #else:
        #    f = ode_model_pressure_2
        f0 = f(t[i], p[i], *pars)
        f1 = f(t[i] + dt, p[i] + dt * f0, *pars)
	    # Increment solution by step size x half of each derivative
        p[i+1] = p[i] + (dt * (0.5 * f0 + 0.5 * f1)) 

    return t, p


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

def plot_pressure_model():

    t, P = improved_euler_pressure(ode_model_pressure_no_mar, t0 = 1980, t1 = 2018, dt = 0.1, p0 = 0.5, pars = [0.05, 0.1])
    plt.plot(t, P)
    plt.title('Pressure')
    plt.xlabel('t')
    plt.ylabel('Pressure (Mpa)')
    plt.show()

def plot_concentration_model():
    year_stock, stock = stock_population()
    year_conc, concentration = nitrate_concentration()

    # fig, ax1 = plt.subplots()
    # plt.title("Stock numbers and concentration against time")
    # ax1.set_xlabel("time [years]")
    # ax1.set_ylabel("Concentration [mg/L]")
    # conc = ax1.scatter(year_conc, concentration, label = "Concentration", color = 'red')

    # ax2 = ax1.twinx()
    # ax2.set_xlabel("time, [years]")
    # ax2.set_ylabel("Stock numbers")
    # stck = ax2.scatter(year_stock, stock, label = "Stock numbers", color = 'green')
    # fig.tight_layout()
    t, C = improved_euler_concentration(ode_model_concentration_no_sink_no_mar, t0 = 1980, t1 = 2018, dt = 0.1, C0 = 0.1, tdelay = 5, pars = [5000000, 0.5, 0.05, 0.3, 0.0001, 0.0003, 0.1, 0.5,0.0003])
    plt.plot(t, C)
    plt.show()

    # plt.annotate(xy=[2010,250000], s='  MAR introduced')
    # plt.plot([2010,2010], [0,700000], color =  'black', linestyle = 'dashed')
    # plt.legend([conc], ["Concentration", "Stock numbers"])
    # plt.show()


def plot_benchmark_concentration():
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
    M = 5000000# mass parameter
    tdelay = 5 #time delay parameter
    P0 = 0.05 #surface pressure parameter
    a = 0.3 #carbon sink infiltration coefficient parameter
    b1 = 0.0001 #infiltration coefficient without carbon sink parameter
    bc = 0.0003 #fresh inflow coefficient
    Pa = 0.1 #pressure at high pressure boundary
    Pmar = 0.05 #pressure due to mar operation
    b = 0.0003 #recharge coefficient


    # Numerical solution
    t, C_Numerical = improved_euler_concentration(ode_model_concentration, t0 = 1980, t1 = 2018, dt = 0.1, C0 = 0.1, pars = [M, t, tdelay, P, P0, a, b1, bc, C, Pa, Pmar, b])

    C_Analytical = np.zeros(len(C_Numerical))
    C_Error = np.zeros(len(C_Numerical))
    inverse_stepsize = np.linspace(1, 3, 21)
    C_Convergence = np.zeros(len(inverse_stepsize))

    for i in range (len(C_Numerical)):
        tmar = 2010
        if t[i] < tmar:
            Pa1 = Pa
        else:
            Pa1 = Pa + Pmar
        tc = 2010
        if t-tdelay < tc:
            b = b1
        else:
            b = a*b1
        n = stock_interpolation(t)
        C_Analytical[i] = math.exp(bc*(P-0.5*Pa1)*C) - (n*b*(P-P0)/(bc*(P-0.5*Pa1)))
        C_Error[i] = abs(C_Analytical[i] - C_Numerical[i])


    for i in range (len(inverse_stepsize)):
        tA, CA = improved_euler_concentration(ode_model_concentration, t0 = 1980, t1 = 2018, dt = inverse_stepsize[i]**(-1), C0 = 0.1, pars = [M, t, tdelay, P, P0, a, b1, bc, C, Pa, Pmar, b])
        C_Convergence[i] = CA[-1]

    plt.subplot(1,3,1)
    plt.plot(t,C_Numerical,'b--',label = 'Numerical')
    plt.plot(t,C_Analytical,'rx',label = 'Analytical')
    plt.legend()
    plt.title('Benchmark')
    plt.xlabel('t')
    plt.ylabel('C')

    plt.subplot(1,3,2)
    plt.plot(t,C_Error,'k-')
    plt.title('Error Analysis')
    plt.xlabel('t')
    plt.ylabel('Relative Error Against Benchmark')

    plt.subplot(1,3,3)
    plt.plot(inverse_stepsize,C_Convergence,'bx')
    plt.title('Timestep Convergence')
    plt.xlabel('1/delta t')
    plt.ylabel('X(t=10)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #ode_model_pressure()
    #ode_model_concentration()
    #stock_population()
    plot_given_data()
    plot_pressure_model()
    plot_concentration_model()




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


# def improved_euler_pressure(f,t0, t1, dt, x0, pars):
#     """Solve an ODE numerically.

#         Parameters:
#         -----------
#         f : callable
#             Function that returns dxdt given variable and parameter inputs.
#         t0 : float
#             Initial time value
#         t1 : float
#             Final time value
#         dt : float
#             Time step used
#         x0 : float
#             Initial value of solution.
#         pars : array-like
#             List of parameters passed to ODE function f.

#         Returns:
#         --------
#         x : array-like
#             Dependent variable solution vector.

#         Notes:
#         ------
#         Assume that ODE function f takes the following inputs, in order:
#             1. independent variable
#             2. dependent variable
#             3. Parameters that will later be passed to the curve fitting function
#             4. all other parameters
#             Refer to the function definitions for order within (3) and (4)
#             5. Optional - counter number to be passed to array parameters
#     """

#     # Allocate return arrays
#     t = np.arange(t0, t1+dt, dt)
#     params_unknown, params_known = pars
#     x = np.zeros(len(t))
#     x[0] = x0
#     i=0
#     # Check if q is iterable or const
#     if isinstance(params_known[0], float) == True:
#         dqdt = 0
#         if len(params_known) != 2:
#             params_known.append(dqdt)
#         # Loop through time values, finding corresponding x value
#         for i in range(0, (len(t) - 1)):
#             # Compute normal euler step
#             x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known)
#             # Corrector step
#             x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known) + f(t[i+1], x_temp, params_unknown, params_known))
#     else:
#         # Get dqdt and append to known parameters
#         dqdt = np.gradient(params_known[0])
#         if len(params_known) != 2:
#             params_known.append(dqdt)
#         # Loop through time values, finding corresponding x value
#         for i in range(0, (len(t) - 1)):
#             # Compute normal euler step
#             x_temp = x[i] + dt*f(t[i], x[i], params_unknown, params_known, i=i)
#             # Corrector step
#             x[i+1] = x[i] + (dt/2)*(f(t[i], x[i], params_unknown, params_known, i=i) + f(t[i+1], x_temp, params_unknown, params_known, i=i))
        
#     return x


# def euler_solve_concentration(f, t0, t1, dt, C0, pars):
    
#     # Allocate return arrays
#     t = np.arange(t0, t1+dt, dt)
#     params_unknown, params_known = pars
#     C = np.zeros(len(t))
#     C[0] = C0

#     for i in range(0, (len(t) - 1)):
        
#         # Compute normal euler step
#         C1 = C[i] + dt*f(t[i], C[i], params_unknown, params_known,i)
        
#         # Corrector step
#         C[i+1] = C[i] + (dt/2)*(f(t[i], C[i], params_unknown, params_known,i) + f(t[i+1], C1, params_unknown, params_known,i))

#     return t, C

#def euler_solve_concentration(f, t0, t1, dt, C0, pars):
    
    # Allocate return arrays
    #t = np.arange(t0, t1+dt, dt)
    #params_unknown, params_known = pars
    #C = np.zeros(len(t))
    #C[0] = C0

    #for i in range(0, (len(t) - 1)):
        
        # Compute normal euler step
        #C1 = C[i] + dt*f(t[i], C[i], params_unknown, params_known,i)
        
        # Corrector step
        #C[i+1] = C[i] + (dt/2)*(f(t[i], C[i], params_unknown, params_known,i) + f(t[i+1], C1, params_unknown, params_known,i))

    #return t, C

#   def plot_benchmark_pressure():
#     ''' Compare analytical and numerical solutions.

#         Parameters:
#         -----------
#         none

#         Returns:
#         --------
#         none

#         Notes:
#         ------
#         This function called within if __name__ == "__main__":

#         It should contain commands to obtain analytical and numerical solutions,
#         plot these, and either display the plot to the screen or save it to the disk.
#     '''
#     M = 5000000# mass parameter
#     tdelay = 5 #time delay parameter
#     P0 = 0.05 #surface pressure parameter
#     a = 0.3 #carbon sink infiltration coefficient parameter
#     b1 = 0.0001 #infiltration coefficient without carbon sink parameter
#     bc = 0.0003 #fresh inflow coefficient
#     Pa = 0.1 #pressure at high pressure boundary
#     Pmar = 0.05 #pressure due to mar operation
#     b = 0.0003 #recharge coefficient


#     # Numerical solution
#     t, P_Numerical = improved_euler_concentration(ode_model_concentration, t0 = 1980, t1 = 2018, dt = 0.1, C0 = 0.1, pars = [M, t, tdelay, P, P0, a, b1, bc, C, Pa, Pmar, b])

#     P_Analytical = np.zeros(len(P_Numerical))
#     P_Error = np.zeros(len(P_Numerical))
#     inverse_stepsize = np.linspace(1, 3, 21)
#     P_Convergence = np.zeros(len(inverse_stepsize))

#     for i in range (len(P_Numerical)):
#         tmar = 2010
#         if t[i] < tmar:
#             Pa1 = Pa
#         else:
#             Pa1 = Pa + Pmar
#         tc = 2010
#         if t-tdelay < tc:
#             b = b1
#         else:
#             b = a*b1
#         P_Analytical[i] = -0.5*Pa*math.exp(-2*b*P) + Pa1/2
#         P_Error[i] = abs(P_Analytical[i] - P_Numerical[i])


#     for i in range (len(inverse_stepsize)):
#         tA, PA = improved_euler_concentration(ode_model_concentration, t0 = 1980, t1 = 2018, dt = inverse_stepsize[i]**(-1), C0 = 0.1, pars = [M, t, tdelay, P, P0, a, b1, bc, C, Pa, Pmar, b])
#         P_Convergence[i] = PA[-1]

#     plt.subplot(1,3,1)
#     plt.plot(t,P_Numerical,'b--',label = 'Numerical')
#     plt.plot(t,P_Analytical,'rx',label = 'Analytical')
#     plt.legend()
#     plt.title('Benchmark')
#     plt.xlabel('t')
#     plt.ylabel('C')

#     plt.subplot(1,3,2)
#     plt.plot(t,P_Error,'k-')
#     plt.title('Error Analysis')
#     plt.xlabel('t')
#     plt.ylabel('Relative Error Against Benchmark')

#     plt.subplot(1,3,3)
#     plt.plot(inverse_stepsize,P_Convergence,'bx')
#     plt.title('Timestep Convergence')
#     plt.xlabel('1/delta t')
#     plt.ylabel('X(t=10)')

#     plt.tight_layout()
#     plt.show()
